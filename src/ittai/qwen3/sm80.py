import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, BFloat16, const_expr
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.memory import SmemAllocator

from .config import *
from .base import MegaBase, acc_mn_view, quad_reduce, gvec


class MegaSm80(MegaBase):

    def __init__(self, tile_n, stages, warps, kv_block, M, Bmax):
        self.bM, self.bN, self.bK = BM, tile_n, 32
        self.stages, self.warps, self.threads, self.kvb = stages, warps, warps * 32, kv_block
        self.atom = (2, warps // 2, 1)
        self.MB, self.Bmax, self.L1 = M // BM, Bmax, L + 1
        self.lanes, self.n_part = tile_n // 8, D // tile_n
        self.red_offs = [self.lanes >> k for k in range(1, int(math.log2(self.lanes)) + 1)]
        self.a_bytes, self.b_bytes = BM * self.bK * 2, tile_n * self.bK * 2
        self.stage_bytes = self.a_bytes + self.b_bytes
        self.pre_bytes = (stages - 1) * self.stage_bytes
        self.q_bytes, self.kv_bytes = BM * HD * 2, kv_block * HD * 2
        self.c_bytes = (BM // 2) * tile_n * 2
        self.attn_pre = self.q_bytes + self.kv_bytes <= self.pre_bytes
        self.smem_bytes = max(stages * self.stage_bytes, self.q_bytes + 2 * self.kv_bytes,
                              self.pre_bytes + max(self.c_bytes, self.q_bytes))
        assert kv_block in (64, 128) and tile_n in (128, 256) and warps in (4, 8)
        assert (warps == 8) or (kv_block == 64), "4 warps: kv block 64"

    @cute.jit
    def __call__(self, h, qkv, attn, act, wqkv, wo, wgu, wd, embed, tokens, pos, cu, stats, s2, qk_w, rope, fin_w, out,
                 tasks, expect, arrive, ready, ctr, meta, grid: Int32, stream: cuda.CUstream):
        dt = BFloat16
        se = self.stage_bytes // 2
        sA_layout = cute.make_layout((self.bM, self.bK, self.stages), stride=(self.bK, 1, se))
        sB_layout = cute.make_layout((self.bN, self.bK, self.stages), stride=(self.bK, 1, se))
        ab_swizzle = cute.make_swizzle(2, 4, 3)
        sC_layout = self._c_layout(self.bN, BM // 2)
        qkv_atom = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1)))
        sQ_layout = cute.tile_to_shape(qkv_atom, (self.bM, HD), (0, 1))
        sKV_layout = cute.tile_to_shape(qkv_atom, (self.kvb, HD), (0, 1))
        atom_async = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), dt, num_bits_per_copy=128)
        atom_sync = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dt, num_bits_per_copy=128)
        copy_AB = self._copy_rows(atom_async, self.bK, self.threads)
        copy_C = self._copy_rows(atom_sync, self.bN, self.threads)
        copy_QKV = self._copy_rows(atom_async, HD, self.threads)
        copy_O = self._copy_rows(atom_sync, HD, self.threads)
        al = self.atom
        mma = cute.make_tiled_mma(warp.MmaF16BF16Op(dt, Float32, (16, 8, 16)), cute.make_layout(al),
                                  permutation_mnk=(al[0] * 16, al[1] * 16, 16))
        mma_attn = cute.make_tiled_mma(warp.MmaF16BF16Op(dt, Float32, (16, 8, 16)), cute.make_layout((self.warps, 1, 1)),
                                       permutation_mnk=(self.warps * 16, 16, 16))
        self.kernel(h, qkv, attn, act, wqkv, wo, wgu, wd, embed, tokens, pos, cu, stats, s2, qk_w, rope, fin_w, out,
                    tasks, expect, arrive, ready, ctr, meta, sA_layout, sB_layout, ab_swizzle, sC_layout, sQ_layout, sKV_layout,
                    copy_AB, copy_C, copy_QKV, copy_O, mma, mma_attn,
                    ).launch(grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, h, qkv, attn, act, wqkv, wo, wgu, wd, embed, tokens, pos, cu, stats, s2, qk_w, rope, fin_w, out,
               tasks, expect, arrive, ready, ctr, meta, sA_layout, sB_layout, ab_swizzle, sC_layout, sQ_layout, sKV_layout,
               copy_AB, copy_C, copy_QKV, copy_O, mma, mma_attn):
        A, W = [h, attn, h, act], [wqkv, wo, wgu, wd]
        tidx, _, _ = cute.arch.thread_idx()
        n_tasks, T = meta[0], meta[1]
        smem = SmemAllocator()
        arena = smem.allocate(self.smem_bytes, byte_alignment=1024)
        s_task = smem.allocate_tensor(Int32, cute.make_layout(2 * TW + 4), byte_alignment=16)
        sA = cute.make_tensor(cute.recast_ptr(arena, ab_swizzle, BFloat16), sA_layout)
        sB = cute.make_tensor(cute.recast_ptr(arena + self.a_bytes, ab_swizzle, BFloat16), sB_layout)
        sC = cute.make_tensor(cute.recast_ptr(arena + self.pre_bytes, dtype=BFloat16), sC_layout)
        sQ = cute.make_tensor(cute.recast_ptr(arena, dtype=BFloat16), sQ_layout)
        sK = cute.make_tensor(cute.recast_ptr(arena + self.q_bytes, dtype=BFloat16), sKV_layout)
        sV = cute.make_tensor(cute.recast_ptr(arena + self.q_bytes + self.kv_bytes, dtype=BFloat16), sKV_layout)
        sO = cute.make_tensor(cute.recast_ptr(arena + self.pre_bytes, dtype=BFloat16), sQ_layout)

        if tidx == 0:
            t = cute.arch.atomic_add(ctr.iterator, Int32(1), sem="relaxed", scope="gpu")
            tn = cute.arch.atomic_add(ctr.iterator, Int32(1), sem="relaxed", scope="gpu")
            tc = cutlass.min(t, n_tasks - 1)
            for w in cutlass.range_constexpr(TW):
                s_task[w] = tasks[tc, w]
            s_task[2 * TW], s_task[2 * TW + 1], s_task[2 * TW + 2] = t, tn, Int32(0)
        cute.arch.sync_threads()
        t, tn, pre = s_task[2 * TW], s_task[2 * TW + 1], Int32(0)
        op, l, i, j = s_task[0], s_task[1], s_task[2], s_task[3]
        d0, c0, d1, c1 = s_task[4], s_task[5], s_task[6], s_task[7]
        while t < n_tasks:
            if op < NGEMM:
                for op_c in cutlass.range_constexpr(NGEMM):
                    if op == op_c:
                        if pre == 0:
                            if tidx == 0:
                                self.spin_ready(ready, d0, c0, d1, c1)
                            cute.arch.sync_threads()
                            self.gemm_prologue(op_c, l, i, j, A[op_c], W[op_c], sA, sB, copy_AB, tidx)
                        self.fetch_next(tasks, ready, ctr, s_task, tn, n_tasks, tidx)
                        acc = self.gemm_mainloop(op_c, l, i, j, A[op_c], W[op_c], sA, sB, copy_AB, mma, tidx)
                        cute.arch.sync_threads()
                        pre = self.prefetch_next(s_task, A, W, sA, sB, sQ, sK, copy_AB, copy_QKV, cu, qkv, ready, n_tasks, tidx)
                        self.gemm_epilogue(op_c, l, i, j, acc, h, qkv, act, s2, stats, qk_w, rope, pos, arrive, ready, expect,
                                           sC, copy_C, mma, tidx)
            elif op == OP_ATTN:
                if pre == 0:
                    if tidx == 0:
                        self.spin_ready(ready, d0, c0, d1, c1)
                    cute.arch.sync_threads()
                    self.attn_prologue(i, j, qkv, cu, sQ, sK, copy_QKV, tidx)
                self.fetch_next(tasks, ready, ctr, s_task, tn, n_tasks, tidx)
                self.run_attn(l, i, j, qkv, attn, cu, s_task, A, W, sA, sB, sQ, sK, sV, sO, copy_AB, copy_QKV, copy_O,
                              mma_attn, arrive, ready, expect, n_tasks, tidx)
                pre = s_task[2 * TW + 2]
            else:
                if tidx == 0:
                    self.spin_ready(ready, d0, c0, d1, c1)
                cute.arch.sync_threads()
                self.fetch_next(tasks, ready, ctr, s_task, tn, n_tasks, tidx)
                if op == OP_EMB:
                    if tidx < BM:
                        self.embed_row(i * BM + tidx, T, embed, tokens, h, stats)
                    cute.arch.sync_threads()
                    if tidx < 32:
                        self.arrive_publish(self.fidx(SLOT_X, 0, i), None, i * BM, s2, stats, arrive, ready, expect)
                else:
                    if tidx < 32:
                        self.final_pool(tidx, i, cu, h, stats, fin_w, out)
                cute.arch.sync_threads()
                pre = self.prefetch_next(s_task, A, W, sA, sB, sQ, sK, copy_AB, copy_QKV, cu, qkv, ready, n_tasks, tidx)
            cute.arch.sync_threads()
            t, tn = tn, s_task[2 * TW + 3]
            op, l, i, j = s_task[TW], s_task[TW + 1], s_task[TW + 2], s_task[TW + 3]
            d0, c0, d1, c1 = s_task[TW + 4], s_task[TW + 5], s_task[TW + 6], s_task[TW + 7]
            cute.arch.sync_threads()

    @cute.jit
    def fetch_next(self, tasks, ready, ctr, s_task, tn: Int32, n_tasks: Int32, tidx: Int32):
        if tidx == 0:
            tn2 = cute.arch.atomic_add(ctr.iterator, Int32(1), sem="relaxed", scope="gpu")
            tc = cutlass.min(tn, n_tasks - 1)
            for w in cutlass.range_constexpr(TW):
                s_task[TW + w] = tasks[tc, w]
            fl = Int32(0)
            if tn < n_tasks:
                if self.deps_bad(ready, s_task[TW + 4], s_task[TW + 5], s_task[TW + 6], s_task[TW + 7]) == 0:
                    fl = Int32(1)
            s_task[2 * TW + 1], s_task[2 * TW + 2], s_task[2 * TW + 3] = tn, fl, tn2

    @cute.jit
    def prefetch_next(self, s_task, A, W, sA, sB, sQ, sK, copy_AB, copy_QKV, cu, qkv, ready, n_tasks: Int32, tidx: Int32) -> Int32:
        if tidx == 0:
            if s_task[2 * TW + 2] == 0:
                if s_task[2 * TW + 1] < n_tasks:
                    if self.deps_bad(ready, s_task[TW + 4], s_task[TW + 5], s_task[TW + 6], s_task[TW + 7]) == 0:
                        s_task[2 * TW + 2] = Int32(1)
        cute.arch.sync_threads()
        pre = s_task[2 * TW + 2]
        opn, ln, in_, jn = s_task[TW], s_task[TW + 1], s_task[TW + 2], s_task[TW + 3]
        if pre != 0:
            if opn < NGEMM:
                for op_n in cutlass.range_constexpr(NGEMM):
                    if opn == op_n:
                        self.gemm_prologue(op_n, ln, in_, jn, A[op_n], W[op_n], sA, sB, copy_AB, tidx)
            elif opn == OP_ATTN:
                if const_expr(self.attn_pre):
                    self.attn_prologue(in_, jn, qkv, cu, sQ, sK, copy_QKV, tidx)
                else:
                    pre = Int32(0)
            else:
                pre = Int32(0)
        return pre

    def _ab_partitions(self, mA, mW, l, i, j, sA, sB, copy_AB, tidx):
        N = cute.size(mW, mode=[0]) // L
        gA = cute.local_tile(mA, (self.bM, self.bK), (i, None))
        gB = cute.local_tile(mW, (self.bN, self.bK), (l * (N // self.bN) + j, None))
        thr = copy_AB.get_slice(tidx)
        return thr.partition_S(gA), thr.partition_D(sA), thr.partition_S(gB), thr.partition_D(sB)

    @cute.jit
    def gemm_prologue(self, op: cutlass.Constexpr, l: Int32, i: Int32, j: Int32, mA, mW, sA, sB, copy_AB, tidx: Int32):
        tAgA, tAsA, tBgB, tBsB = self._ab_partitions(mA, mW, l, i, j, sA, sB, copy_AB, tidx)
        for s in cutlass.range_constexpr(self.stages - 1):
            cute.copy(copy_AB, tAgA[None, None, None, s], tAsA[None, None, None, s])
            cute.copy(copy_AB, tBgB[None, None, None, s], tBsB[None, None, None, s])
            cute.arch.cp_async_commit_group()

    @cute.jit
    def gemm_mainloop(self, op: cutlass.Constexpr, l: Int32, i: Int32, j: Int32, mA, mW, sA, sB, copy_AB, mma, tidx: Int32):
        tAgA, tAsA, tBgB, tBsB = self._ab_partitions(mA, mW, l, i, j, sA, sB, copy_AB, tidx)
        ns = self.stages
        k_tiles = SPEC[op]["K"] // self.bK
        k_idx = Int32(ns - 1)
        thr_mma = mma.get_slice(tidx)
        tCsA, tCsB = thr_mma.partition_A(sA), thr_mma.partition_B(sB)
        tCrA = mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = mma.make_fragment_B(tCsB[None, None, None, 0])
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.bM, self.bN)), Float32)
        acc.fill(0.0)
        atom_s2r = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16)
        s2r_A, s2r_B = cute.make_tiled_copy_A(atom_s2r, mma), cute.make_tiled_copy_B(atom_s2r, mma)
        thr_A, thr_B = s2r_A.get_slice(tidx), s2r_B.get_slice(tidx)
        tCsA_cv, tCrA_cv = thr_A.partition_S(sA), thr_A.retile(tCrA)
        tCsB_cv, tCrB_cv = thr_B.partition_S(sB), thr_B.retile(tCrB)
        rd = Int32(0)
        wr = Int32(ns - 1)
        nkb = cute.size(tCrA, mode=[2])
        cute.arch.cp_async_wait_group(ns - 2)
        cute.arch.sync_threads()
        cute.copy(s2r_A, tCsA_cv[None, None, 0, rd], tCrA_cv[None, None, 0])
        cute.copy(s2r_B, tCsB_cv[None, None, 0, rd], tCrB_cv[None, None, 0])
        for kt in cutlass.range(k_tiles, unroll=1):
            for kb in cutlass.range_constexpr(nkb):
                if kb == nkb - 1:
                    cute.arch.cp_async_wait_group(ns - 2)
                    cute.arch.sync_threads()
                    rd = rd + 1
                    if rd == ns:
                        rd = Int32(0)
                kn = (kb + 1) % nkb
                cute.copy(s2r_A, tCsA_cv[None, None, kn, rd], tCrA_cv[None, None, kn])
                cute.copy(s2r_B, tCsB_cv[None, None, kn, rd], tCrB_cv[None, None, kn])
                if kb == 0:
                    if kt + ns - 1 < k_tiles:
                        cute.copy(copy_AB, tAgA[None, None, None, k_idx], tAsA[None, None, None, wr])
                        cute.copy(copy_AB, tBgB[None, None, None, k_idx], tBsB[None, None, None, wr])
                    k_idx = k_idx + 1
                    cute.arch.cp_async_commit_group()
                    wr = wr + 1
                    if wr == ns:
                        wr = Int32(0)
                cute.gemm(mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        return acc

    @cute.jit
    def gemm_epilogue(self, op: cutlass.Constexpr, l: Int32, i: Int32, j: Int32, acc, h, qkv, act, s2, stats, qk_w, rope, pos,
                      arrive, ready, expect, sC, copy_C, mma, tidx: Int32):
        spec = SPEC[op]
        row0, col0 = i * self.bM, j * self.bN
        thr_mma = mma.get_slice(tidx)
        tCsC = thr_mma.partition_C(sC)
        tCrD = cute.make_fragment_like(acc, BFloat16)
        if const_expr(spec["s_in"] is not None):
            tCcC_mn = acc_mn_view(thr_mma.partition_C(cute.make_identity_tensor((self.bM, self.bN))))
            acc_mn, out_mn = acc_mn_view(acc), acc_mn_view(tCrD)
            for r in cutlass.range_constexpr(cute.size(tCcC_mn.shape[0])):
                rstd = stats[spec["s_in"], row0 + tCcC_mn[r, 0][0]]
                for c in cutlass.range_constexpr(cute.size(tCcC_mn.shape[1])):
                    out_mn[r, c] = (acc_mn[r, c] * rstd).to(BFloat16)
        else:
            tCrD.store(acc.load().to(BFloat16))
        tCrD_div = cute.logical_divide(tCrD, (None, 2, None))
        thr_C = copy_C.get_slice(tidx)
        tCsC_epi = thr_C.partition_S(sC)
        lane = tidx % self.lanes
        rows_per_step = self.threads // self.lanes
        col = col0 + lane * 8
        for hh in cutlass.range_constexpr(2):
            cute.autovec_copy(tCrD_div[(None, (None, hh), None)], tCsC)
            cute.arch.sync_threads()
            for m in cutlass.range_constexpr(cute.size(tCsC_epi, mode=[1])):
                row = row0 + 64 * hh + tidx // self.lanes + rows_per_step * m
                v8 = cute.make_rmem_tensor(cute.make_layout(8), BFloat16)
                cute.autovec_copy(tCsC_epi[None, m, None], v8)
                self.epi_chunk(spec["kind"], spec["s_out"], l, j, row, col, lane, v8, h, qkv, act, s2, qk_w, rope, pos)
            cute.arch.sync_threads()
        if tidx < 32:
            self.arrive_publish(self.fidx(spec["out"], l + spec["dl"], i), spec["s_out"], row0, s2, stats, arrive, ready, expect)

    @cute.jit
    def attn_prologue(self, i: Int32, j: Int32, qkv, cu, sQ, sK, copy_QKV, tidx: Int32):
        s, g = i // NKV, i % NKV
        start = cu[s]
        thr = copy_QKV.get_slice(tidx)
        tQsQ = thr.partition_D(sQ)
        nq = cute.size(tQsQ, mode=[1]) // G
        for hh in cutlass.range_constexpr(G):
            gQ = cute.local_tile(cute.domain_offset((start + j * QB, 0), qkv), (QB, HD), (0, G * g + hh))
            tQgQ = thr.partition_S(gQ)
            for m in cutlass.range_constexpr(nq):
                cute.copy(copy_QKV, tQgQ[None, m, None], tQsQ[None, hh * nq + m, None])
        nb = (j * QB + QB + self.kvb - 1) // self.kvb
        gK = cute.local_tile(cute.domain_offset((start, 0), qkv), (self.kvb, HD), (nb - 1, K_OFF // HD + g))
        cute.copy(copy_QKV, thr.partition_S(gK), thr.partition_D(sK))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def run_attn(self, l: Int32, i: Int32, j: Int32, qkv, attn, cu, s_task, A, W, sA, sB, sQ, sK, sV, sO, copy_AB, copy_QKV,
                 copy_O, mma, arrive, ready, expect, n_tasks: Int32, tidx: Int32):
        s, g = i // NKV, i % NKV
        start, Ls = cu[s], cu[s + 1] - cu[s]
        n_blocks = (j * QB + QB + self.kvb - 1) // self.kvb
        thr = copy_QKV.get_slice(tidx)
        gKV = cute.local_tile(cute.domain_offset((start, 0), qkv), (self.kvb, HD), (None, None))
        tKgK = thr.partition_S(gKV)
        tKsK, tVsV = thr.partition_D(sK), thr.partition_D(sV)
        thr_mma = mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        sVt = cute.composition(sV, cute.make_layout((HD, self.kvb), stride=(self.kvb, 1)))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.bM, HD)), Float32)
        acc_O.fill(0.0)
        s2r_Q = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16), mma)
        s2r_K = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16), mma)
        s2r_V = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), BFloat16), mma)
        thr_Q, thr_K, thr_V = s2r_Q.get_slice(tidx), s2r_K.get_slice(tidx), s2r_V.get_slice(tidx)
        tSsQ, tSrQ_cv = thr_Q.partition_S(sQ), thr_Q.retile(tSrQ)
        tSsK, tSrK_cv = thr_K.partition_S(sK), thr_K.retile(tSrK)
        tOsVt, tOrVt_cv = thr_V.partition_S(sVt), thr_V.retile(tOrVt)
        nrow = acc_O.shape[0][0] * acc_O.shape[1]
        row_max = cute.make_rmem_tensor(cute.make_layout(nrow), Float32)
        row_sum = cute.make_rmem_tensor(cute.make_layout(nrow), Float32)
        row_max.fill(-Float32.inf)
        row_sum.fill(0.0)
        k_col, v_col = K_OFF // HD + g, V_OFF // HD + g
        tScS_mn = acc_mn_view(thr_mma.partition_C(cute.make_identity_tensor((self.bM, self.kvb))))
        qpos = [j * QB + tScS_mn[r, 0][0] % QB for r in range(nrow)]
        kcol = [tScS_mn[0, c][1] for c in range(cute.size(tScS_mn.shape[1]))]
        for nb in cutlass.range(n_blocks, unroll=1):
            n_block = n_blocks - 1 - nb
            acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.bM, self.kvb)), Float32)
            acc_S.fill(0.0)
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            cute.copy(copy_QKV, tKgK[None, None, None, n_block, v_col], tVsV)
            cute.arch.cp_async_commit_group()
            nk = cute.size(tSsQ, mode=[2])
            cute.copy(s2r_Q, tSsQ[None, None, 0], tSrQ_cv[None, None, 0])
            cute.copy(s2r_K, tSsK[None, None, 0], tSrK_cv[None, None, 0])
            for k in cutlass.range_constexpr(nk):
                kn = (k + 1) % nk
                cute.copy(s2r_Q, tSsQ[None, None, kn], tSrQ_cv[None, None, kn])
                cute.copy(s2r_K, tSsK[None, None, kn], tSrK_cv[None, None, kn])
                cute.gemm(mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            if n_block > 0:
                cute.copy(copy_QKV, tKgK[None, None, None, n_block - 1, k_col], tKsK)
                cute.arch.cp_async_commit_group()
            acc_S_mn, acc_O_mn = acc_mn_view(acc_S), acc_mn_view(acc_O)
            for r in cutlass.range_constexpr(nrow):
                if nb == 0:
                    for c in cutlass.range_constexpr(len(kcol)):
                        if n_block * self.kvb + kcol[c] > qpos[r]:
                            acc_S_mn[r, c] = -Float32.inf
                row = acc_S_mn[r, None].load()
                mx = quad_reduce(row.reduce(cute.ReductionOp.MAX, -Float32.inf, 0), lambda x, y: cute.arch.fmax(x, y))
                mx = cute.arch.fmax(row_max[r], mx)
                row_exp = cute.math.exp2(row * Float32(SCALE_LOG2) - mx * Float32(SCALE_LOG2), fastmath=True)
                corr = cute.math.exp2(row_max[r] * Float32(SCALE_LOG2) - mx * Float32(SCALE_LOG2), fastmath=True)
                row_sum[r] = row_exp.reduce(cute.ReductionOp.ADD, Float32.zero, 0) + row_sum[r] * corr
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * corr
                row_max[r] = mx
                acc_S_mn[r, None] = row_exp
            rP = cute.make_fragment_like(acc_S, BFloat16)
            rP.store(acc_S.load().to(BFloat16))
            ld = cute.logical_divide(rP.layout, (None, None, 2))
            tOrS = cute.make_tensor(rP.iterator, cute.make_layout(
                ((ld.shape[0], ld.shape[2][0]), ld.shape[1], ld.shape[2][1]),
                stride=((ld.stride[0], ld.stride[2][0]), ld.stride[1], ld.stride[2][1])))
            nkv = cute.size(tOrS.shape[2])
            cute.copy(s2r_V, tOsVt[None, None, 0], tOrVt_cv[None, None, 0])
            for k in cutlass.range_constexpr(nkv):
                kn = (k + 1) % nkv
                cute.copy(s2r_V, tOsVt[None, None, kn], tOrVt_cv[None, None, kn])
                cute.gemm(mma, acc_O, tOrS[None, None, k], tOrVt[None, None, k], acc_O)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        pre = self.prefetch_next(s_task, A, W, sA, sB, sQ, sK, copy_AB, copy_QKV, cu, qkv, ready, n_tasks, tidx)
        if tidx == 0:
            s_task[2 * TW + 2] = pre
        acc_O_mn = acc_mn_view(acc_O)
        for r in cutlass.range_constexpr(nrow):
            rs = quad_reduce(row_sum[r], lambda x, y: x + y)
            inv = Float32(1.0) / rs
            if rs == 0.0:
                inv = Float32(1.0)
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv
        rO = cute.make_fragment_like(acc_O, BFloat16)
        rO.store(acc_O.load().to(BFloat16))
        thr_O = cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16), mma).get_slice(tidx)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16), thr_O.retile(rO), thr_O.partition_D(sO))
        cute.arch.sync_threads()
        thr_o = copy_O.get_slice(tidx)
        tOsO = thr_o.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO)
        cute.autovec_copy(tOsO, tOrO)
        nq = cute.size(tOsO, mode=[1]) // G
        rps = self.threads // (HD // 8)
        for hh in cutlass.range_constexpr(G):
            for m in cutlass.range_constexpr(nq):
                rq = j * QB + tidx // (HD // 8) + rps * m
                if rq < Ls:
                    cute.autovec_copy(tOrO[None, hh * nq + m, None],
                                      gvec(attn.iterator, (start + rq) * DQ + (G * g + hh) * HD + (tidx % (HD // 8)) * 8, BFloat16, 8))
        cute.arch.sync_threads()
        if tidx < 32:
            self.attn_publish(l, s, j, cu, arrive, ready, expect)

    @cute.jit
    def attn_publish(self, l: Int32, s: Int32, j: Int32, cu, arrive, ready, expect):
        q0 = cu[s] + j * QB
        q1 = cutlass.min(q0 + QB, cu[s + 1]) - 1
        self.arrive_publish(self.fidx(SLOT_ATTN, l, q0 // BM), None, Int32(0), None, None, arrive, ready, expect)
        if q1 // BM != q0 // BM:
            self.arrive_publish(self.fidx(SLOT_ATTN, l, q1 // BM), None, Int32(0), None, None, arrive, ready, expect)
        self.arrive_publish(NSLOT * self.L1 * self.MB + l * self.Bmax + s, None, Int32(0), None, None, arrive, ready, expect)