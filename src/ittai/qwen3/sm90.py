import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, BFloat16, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.memory import SmemAllocator

from .config import *
from .base import MegaBase, acc_mn_view, quad_reduce, gvec


class MegaSm90(MegaBase):

    def __init__(self, tile_n, stages, kv_block, M, Bmax):
        self.bM, self.bN, self.bK = BM, tile_n, 64
        self.stages, self.kvb, self.threads, self.warps = stages, kv_block, 384, 12
        self.MB, self.Bmax, self.L1 = M // BM, Bmax, L + 1
        self.lanes, self.n_part = tile_n // 8, D // tile_n
        self.red_offs = [self.lanes >> k for k in range(1, int(math.log2(self.lanes)) + 1)]
        self.a_bytes, self.b_bytes = BM * self.bK * 2, tile_n * self.bK * 2
        self.q_bytes, self.kv_bytes = QB * HD * 2 * G, kv_block * HD * 2
        self.c_bytes = BM * tile_n * 2
        self.arena = max(stages * (self.a_bytes + self.b_bytes), 2 * 2 * self.kv_bytes + self.q_bytes)
        self.smem_bytes = self.arena + self.c_bytes
        self.gemm_tx, self.kv_tx, self.q_tx = self.a_bytes + self.b_bytes, 2 * self.kv_bytes, self.q_bytes
        self.nbars = 2 * stages + 2 * 2 + 2 + 2 * 2
        assert kv_block in (64, 128) and tile_n in (128, 256)

    @cute.jit
    def __call__(self, h, qkv, attn, act, wqkv, wo, wgu, wd, embed, tokens, pos, cu, stats, s2, qk_w, rope, fin_w, out,
                 tasks, expect, arrive, ready, ctr, meta, grid: Int32, stream: cuda.CUstream):
        import cutlass.utils.hopper_helpers as sm90
        from cutlass.cute.nvgpu.warpgroup import OperandMajorMode, OperandSource
        from cutlass.utils import LayoutEnum
        km = LayoutEnum.from_tensor(h)
        mma = sm90.make_trivial_tiled_mma(BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K, Float32, (2, 1, 1), (64, self.bN))
        qk_mma = sm90.make_trivial_tiled_mma(BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.K, Float32, (1, 1, 1), (QB, self.kvb))
        pv_mma = sm90.make_trivial_tiled_mma(BFloat16, BFloat16, OperandMajorMode.K, OperandMajorMode.MN, Float32, (1, 1, 1),
                                             (QB, HD), OperandSource.RMEM)
        a_sl = sm90.make_smem_layout_a(km, (self.bM, self.bN, self.bK), BFloat16, self.stages)
        b_sl = sm90.make_smem_layout_b(km, (self.bM, self.bN, self.bK), BFloat16, self.stages)
        q_sl = sm90.make_smem_layout_a(km, (QB, self.kvb, HD), BFloat16, G)
        k_sl = sm90.make_smem_layout_b(km, (QB, self.kvb, HD), BFloat16, 2)
        v_view = cute.make_tensor(qkv.iterator, cute.select(qkv.layout, mode=[1, 0]))
        v_sl = sm90.make_smem_layout_b(LayoutEnum.from_tensor(v_view), (QB, HD, self.kvb), BFloat16, 2)
        g2s = cpasync.CopyBulkTensorTileG2SOp()
        a1, b1 = cute.slice_(a_sl, (None, None, 0)), cute.slice_(b_sl, (None, None, 0))
        tma_a = [cpasync.make_tiled_tma_atom(g2s, x, a1, (self.bM, self.bK)) for x in (h, attn, h, act)]
        tma_b = [cpasync.make_tiled_tma_atom(g2s, x, b1, (self.bN, self.bK)) for x in (wqkv, wo, wgu, wd)]
        tma_q = cpasync.make_tiled_tma_atom(g2s, qkv, cute.slice_(q_sl, (None, None, 0)), (QB, HD))
        tma_k = cpasync.make_tiled_tma_atom(g2s, qkv, cute.slice_(k_sl, (None, None, 0)), (self.kvb, HD))
        tma_v = cpasync.make_tiled_tma_atom(g2s, v_view, cute.slice_(v_sl, (None, None, 0)), (HD, self.kvb))
        atom_sync = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        copy_C = self._copy_rows(atom_sync, self.bN, 128)
        copy_O = self._copy_rows(atom_sync, HD, 128)
        sC_layout = self._c_layout(self.bN)
        self.kernel(tma_a[0][0], tma_a[0][1], tma_a[1][0], tma_a[1][1], tma_a[2][0], tma_a[2][1], tma_a[3][0], tma_a[3][1],
                    tma_b[0][0], tma_b[0][1], tma_b[1][0], tma_b[1][1], tma_b[2][0], tma_b[2][1], tma_b[3][0], tma_b[3][1],
                    tma_q[0], tma_q[1], tma_k[0], tma_k[1], tma_v[0], tma_v[1],
                    mma, qk_mma, pv_mma, a_sl, b_sl, q_sl, k_sl, v_sl, sC_layout, copy_C, copy_O,
                    h, qkv, attn, act, embed, tokens, pos, cu, stats, s2, qk_w, rope, fin_w, out,
                    tasks, expect, arrive, ready, ctr, meta,
                    ).launch(grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, ta0, ga0, ta1, ga1, ta2, ga2, ta3, ga3, tb0, gb0, tb1, gb1, tb2, gb2, tb3, gb3,
               tma_q, gq, tma_k, gk, tma_v, gv, mma, qk_mma, pv_mma, a_sl, b_sl, q_sl, k_sl, v_sl, sC_layout, copy_C, copy_O,
               h, qkv, attn, act, embed, tokens, pos, cu, stats, s2, qk_w, rope, fin_w, out, tasks, expect, arrive, ready, ctr, meta):
        TA, GA, TB, GB = [ta0, ta1, ta2, ta3], [ga0, ga1, ga2, ga3], [tb0, tb1, tb2, tb3], [gb0, gb1, gb2, gb3]
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg = cute.arch.make_warp_uniform(tidx // 128)
        n_tasks, T = meta[0], meta[1]
        smem = SmemAllocator()
        bars = smem.allocate_tensor(Int64, cute.make_layout(self.nbars), byte_alignment=8)
        s_task = smem.allocate_tensor(Int32, cute.make_layout((2, TW), stride=(TW, 1)), byte_alignment=16)
        arena = smem.allocate(self.arena, byte_alignment=1024)
        sC = cute.make_tensor(cute.recast_ptr(smem.allocate(self.c_bytes, byte_alignment=1024), dtype=BFloat16), sC_layout)
        g_full = bars.iterator
        g_empty = g_full + self.stages
        kv_full, kv_empty = g_empty + self.stages, g_empty + self.stages + 2
        q_full, q_empty = kv_empty + 2, kv_empty + 3
        i_full, i_empty = q_empty + 1, q_empty + 3
        sA = cute.make_tensor(cute.recast_ptr(arena, a_sl.inner, BFloat16), a_sl.outer)
        sB = cute.make_tensor(cute.recast_ptr(arena + self.stages * self.a_bytes, b_sl.inner, BFloat16), b_sl.outer)
        sK = cute.make_tensor(cute.recast_ptr(arena, k_sl.inner, BFloat16), k_sl.outer)
        sV = cute.make_tensor(cute.recast_ptr(arena + 2 * self.kv_bytes, v_sl.inner, BFloat16), v_sl.outer)
        sQ = cute.make_tensor(cute.recast_ptr(arena + 4 * self.kv_bytes, q_sl.inner, BFloat16), q_sl.outer)
        if warp_idx == 0:
            for x in TA + TB + [tma_q, tma_k, tma_v]:
                cpasync.prefetch_descriptor(x)
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(g_full + s, 1)
                    cute.arch.mbarrier_init(g_empty + s, 2)
                for s in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(kv_full + s, 1)
                    cute.arch.mbarrier_init(kv_empty + s, 2)
                    cute.arch.mbarrier_init(i_full + s, 1)
                    cute.arch.mbarrier_init(i_empty + s, 2)
                cute.arch.mbarrier_init(q_full, 1)
                cute.arch.mbarrier_init(q_empty, 2)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        sA_g, sB_g = cute.group_modes(sA, 0, 2), cute.group_modes(sB, 0, 2)
        tAs, tAg, tBs, tBg = [], [], [], []
        for o in cutlass.range_constexpr(NGEMM):
            a_s, a_g = cpasync.tma_partition(TA[o], 0, cute.make_layout(1), sA_g,
                                             cute.group_modes(cute.local_tile(GA[o], (self.bM, self.bK), (None, None)), 0, 2))
            b_s, b_g = cpasync.tma_partition(TB[o], 0, cute.make_layout(1), sB_g,
                                             cute.group_modes(cute.local_tile(GB[o], (self.bN, self.bK), (None, None)), 0, 2))
            tAs.append(a_s); tAg.append(a_g); tBs.append(b_s); tBg.append(b_g)
        if wg == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                self.producer(TA, tAg, tAs, TB, tBg, tBs, tma_q, gq, tma_k, gk, tma_v, gv, sQ, sK, sV,
                              g_full, g_empty, kv_full, kv_empty, q_full, q_empty, i_full, i_empty,
                              s_task, tasks, ready, ctr, cu, n_tasks)
        else:
            cute.arch.setmaxregister_increase(232)
            self.consumer(wg - 1, tidx % 128, warp_idx % 4, T, mma, qk_mma, pv_mma, sA, sB, sQ, sK, sV, sC, copy_C, copy_O,
                          g_full, g_empty, kv_full, kv_empty, q_full, q_empty, i_full, i_empty, s_task,
                          h, qkv, attn, act, embed, tokens, pos, cu, stats, s2, qk_w, rope, fin_w, out, arrive, ready, expect)

    @cute.jit
    def producer(self, TA, tAg, tAs, TB, tBg, tBs, tma_q, gq, tma_k, gk, tma_v, gv, sQ, sK, sV,
                 g_full, g_empty, kv_full, kv_empty, q_full, q_empty, i_full, i_empty, s_task, tasks, ready, ctr, cu, n_tasks):
        n = Int32(0)
        prev_attn = Int32(-1)
        gs, gp, ks, kp, qp = Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
        running = Boolean(True)
        while running:
            slot, use = n % 2, n // 2
            cute.arch.mbarrier_wait(i_empty + slot, (use % 2) ^ 1)
            with cute.arch.elect_one():
                self.fetch_task(ctr, tasks, ready, s_task, slot, n_tasks)
                cute.arch.mbarrier_arrive(i_full + slot)
            cute.arch.sync_warp()
            cute.arch.fence_proxy("async.global")
            op, l, i, j = s_task[slot, 0], s_task[slot, 1], s_task[slot, 2], s_task[slot, 3]
            if op < 0:
                running = Boolean(False)
            else:
                is_attn = Int32(0)
                if op == OP_ATTN:
                    is_attn = Int32(1)
                if n > 0:
                    if is_attn != prev_attn:
                        cute.arch.mbarrier_wait(i_empty + (n - 1) % 2, ((n - 1) // 2) % 2)
                prev_attn = is_attn
                for op_c in cutlass.range_constexpr(NGEMM):
                    if op == op_c:
                        N = SPEC[op_c]["N"]
                        for kt in cutlass.range(SPEC[op_c]["K"] // self.bK, unroll=1):
                            cute.arch.mbarrier_wait(g_empty + gs, gp ^ 1)
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(g_full + gs, self.gemm_tx)
                            cute.copy(TA[op_c], tAg[op_c][(None, i, kt)], tAs[op_c][(None, gs)], tma_bar_ptr=g_full + gs)
                            cute.copy(TB[op_c], tBg[op_c][(None, l * (N // self.bN) + j, kt)], tBs[op_c][(None, gs)], tma_bar_ptr=g_full + gs)
                            gs = gs + 1
                            if gs == self.stages:
                                gs, gp = Int32(0), gp ^ 1
                if op == OP_ATTN:
                    s, g = i // NKV, i % NKV
                    start = cu[s]
                    gQ = cute.local_tile(cute.domain_offset((start + j * QB, 0), gq), (QB, HD), (None, None))
                    tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
                    cute.arch.mbarrier_wait(q_empty, qp ^ 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(q_full, self.q_tx)
                    for hh in cutlass.range_constexpr(G):
                        cute.copy(tma_q, tQg[(None, 0, G * g + hh)], tQs[(None, hh)], tma_bar_ptr=q_full)
                    qp = qp ^ 1
                    gK = cute.local_tile(cute.domain_offset((start, 0), gk), (self.kvb, HD), (None, None))
                    gV = cute.local_tile(cute.domain_offset((0, start), gv), (HD, self.kvb), (None, None))
                    tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
                    tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))
                    n_blocks = (j * QB + QB + self.kvb - 1) // self.kvb
                    for nb in cutlass.range(n_blocks, unroll=1):
                        blk = n_blocks - 1 - nb
                        cute.arch.mbarrier_wait(kv_empty + ks, kp ^ 1)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(kv_full + ks, self.kv_tx)
                        cute.copy(tma_k, tKg[(None, blk, K_OFF // HD + g)], tKs[(None, ks)], tma_bar_ptr=kv_full + ks)
                        cute.copy(tma_v, tVg[(None, V_OFF // HD + g, blk)], tVs[(None, ks)], tma_bar_ptr=kv_full + ks)
                        ks = ks + 1
                        if ks == 2:
                            ks, kp = Int32(0), kp ^ 1
                n = n + 1

    @cute.jit
    def fetch_task(self, ctr, tasks, ready, s_task, slot: Int32, n_tasks: Int32):
        t = cute.arch.atomic_add(ctr.iterator, Int32(1), sem="relaxed", scope="gpu")
        if t < n_tasks:
            for w in cutlass.range_constexpr(TW):
                s_task[slot, w] = tasks[t, w]
            self.spin_ready(ready, s_task[slot, 4], s_task[slot, 5], s_task[slot, 6], s_task[slot, 7])
        else:
            s_task[slot, 0] = Int32(-1)

    @cute.jit
    def consumer(self, cidx, tid, wiw, T, mma, qk_mma, pv_mma, sA, sB, sQ, sK, sV, sC, copy_C, copy_O,
                 g_full, g_empty, kv_full, kv_empty, q_full, q_empty, i_full, i_empty, s_task,
                 h, qkv, attn, act, embed, tokens, pos, cu, stats, s2, qk_w, rope, fin_w, out, arrive, ready, expect):
        thr_mma = mma.get_slice(cidx * 128)
        tCrA = mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = mma.make_fragment_B(thr_mma.partition_B(sB))
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.bM, self.bN)), Float32)
        mma.set(warpgroup.Field.ACCUMULATE, True)
        qk_mma.set(warpgroup.Field.ACCUMULATE, True)
        pv_mma.set(warpgroup.Field.ACCUMULATE, True)
        n = Int32(0)
        gs, gp, ks, kp, qp = Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
        running = Boolean(True)
        while running:
            slot, use = n % 2, n // 2
            cute.arch.mbarrier_wait(i_full + slot, use % 2)
            op, l, i, j = s_task[slot, 0], s_task[slot, 1], s_task[slot, 2], s_task[slot, 3]
            if op < 0:
                running = Boolean(False)
            else:
                for op_c in cutlass.range_constexpr(NGEMM):
                    if op == op_c:
                        gs, gp = self.mainloop(op_c, mma, tCrA, tCrB, acc, g_full, g_empty, gs, gp, wiw)
                        self.gemm_epilogue(op_c, cidx, tid, l, i, j, acc, h, qkv, act, s2, stats, qk_w, rope, pos, arrive, ready,
                                           expect, sC, copy_C, mma)
                if op == OP_ATTN:
                    ks, kp, qp = self.attention(cidx, tid, wiw, l, i, j, qk_mma, pv_mma, sQ, sK, sV, kv_full, kv_empty, q_full,
                                                q_empty, ks, kp, qp, cu, attn, copy_O, arrive, ready, expect)
                elif op == OP_EMB:
                    if tid % 2 == 0:
                        self.embed_row(i * BM + cidx * 64 + tid // 2, T, embed, tokens, h, stats)
                    cute.arch.fence_proxy("async.global")
                    self.wg_sync(cidx)
                    if wiw == 0:
                        self.arrive_publish(self.fidx(SLOT_X, 0, i), None, i * BM, s2, stats, arrive, ready, expect)
                elif op == OP_FINAL:
                    if cidx == 0:
                        if wiw == 0:
                            self.final_pool(tid, i, cu, h, stats, fin_w, out)
                self.wg_sync(cidx)
                if wiw == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(i_empty + slot)
                n = n + 1

    @cute.jit
    def wg_sync(self, cidx):
        if cidx == 0:
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
        else:
            cute.arch.barrier(barrier_id=2, number_of_threads=128)

    @cute.jit
    def mainloop(self, op: cutlass.Constexpr, mma, tCrA, tCrB, acc, g_full, g_empty, gs, gp, wiw):
        acc.fill(0.0)
        rel = gs
        nkb = cute.size(tCrA, mode=[2])
        for kt in cutlass.range(SPEC[op]["K"] // self.bK, unroll=1):
            cute.arch.mbarrier_wait(g_full + gs, gp)
            warpgroup.fence()
            for kb in cutlass.range_constexpr(nkb):
                cute.gemm(mma, acc, tCrA[(None, None, kb, gs)], tCrB[(None, None, kb, gs)], acc)
            warpgroup.commit_group()
            warpgroup.wait_group(1)
            if kt > 0:
                if wiw == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(g_empty + rel)
                rel = rel + 1
                if rel == self.stages:
                    rel = Int32(0)
            gs = gs + 1
            if gs == self.stages:
                gs, gp = Int32(0), gp ^ 1
        warpgroup.wait_group(0)
        if wiw == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(g_empty + rel)
        return gs, gp

    @cute.jit
    def gemm_epilogue(self, op: cutlass.Constexpr, cidx, tid, l, i, j, acc, h, qkv, act, s2, stats, qk_w, rope, pos,
                      arrive, ready, expect, sC, copy_C, mma):
        spec = SPEC[op]
        row0, col0 = i * self.bM, j * self.bN
        thr_mma = mma.get_slice(cidx * 128 + tid)
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
        cute.autovec_copy(tCrD, thr_mma.partition_C(sC))
        self.wg_sync(cidx)
        sC_wg = cute.local_tile(sC, (64, self.bN), (cidx, 0))
        thr_C = copy_C.get_slice(tid)
        tCsC_epi = thr_C.partition_S(sC_wg)
        lane = tid % self.lanes
        rps = 128 // self.lanes
        col = col0 + lane * 8
        for m in cutlass.range_constexpr(cute.size(tCsC_epi, mode=[1])):
            row = row0 + cidx * 64 + tid // self.lanes + rps * m
            v8 = cute.make_rmem_tensor(cute.make_layout(8), BFloat16)
            cute.autovec_copy(tCsC_epi[None, m, None], v8)
            self.epi_chunk(spec["kind"], spec["s_out"], l, j, row, col, lane, v8, h, qkv, act, s2, qk_w, rope, pos)
        cute.arch.fence_proxy("async.global")
        self.wg_sync(cidx)
        if tid < 32:
            self.arrive_publish(self.fidx(spec["out"], l + spec["dl"], i), spec["s_out"], row0, s2, stats, arrive, ready, expect)

    @cute.jit
    def attention(self, cidx, tid, wiw, l, i, j, qk_mma, pv_mma, sQ, sK, sV, kv_full, kv_empty, q_full, q_empty, ks, kp, qp,
                  cu, attn, copy_O, arrive, ready, expect):
        s, g = i // NKV, i % NKV
        start, Ls = cu[s], cu[s + 1] - cu[s]
        n_blocks = (j * QB + QB + self.kvb - 1) // self.kvb
        qk_thr, pv_thr = qk_mma.get_slice(tid), pv_mma.get_slice(tid)
        tSrQ = qk_thr.make_fragment_A(qk_thr.partition_A(sQ))
        tSrK = qk_thr.make_fragment_B(qk_thr.partition_B(sK))
        tOrV = pv_thr.make_fragment_B(pv_thr.partition_B(sV))
        acc_S = cute.make_rmem_tensor(qk_thr.partition_shape_C((QB, self.kvb)), Float32)
        acc_O = cute.make_rmem_tensor(pv_thr.partition_shape_C((QB, HD)), Float32)
        acc_O.fill(0.0)
        tScS_mn = acc_mn_view(qk_thr.partition_C(cute.make_identity_tensor((QB, self.kvb))))
        nrow = cute.size(tScS_mn.shape[0])
        qpos = [j * QB + tScS_mn[r, 0][0] for r in range(nrow)]
        kcol = [tScS_mn[0, c][1] for c in range(cute.size(tScS_mn.shape[1]))]
        row_max = cute.make_rmem_tensor(cute.make_layout(nrow), Float32)
        row_sum = cute.make_rmem_tensor(cute.make_layout(nrow), Float32)
        row_max.fill(-Float32.inf)
        row_sum.fill(0.0)
        cute.arch.mbarrier_wait(q_full, qp)
        qp = qp ^ 1
        for nb in cutlass.range(n_blocks, unroll=1):
            cute.arch.mbarrier_wait(kv_full + ks, kp)
            acc_S.fill(0.0)
            warpgroup.fence()
            for kb in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
                cute.gemm(qk_mma, acc_S, tSrQ[(None, None, kb, cidx)], tSrK[(None, None, kb, ks)], acc_S)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            acc_S_mn, acc_O_mn = acc_mn_view(acc_S), acc_mn_view(acc_O)
            for r in cutlass.range_constexpr(nrow):
                if nb == 0:
                    for c in cutlass.range_constexpr(len(kcol)):
                        if (n_blocks - 1) * self.kvb + kcol[c] > qpos[r]:
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
            tOrP = self.acc_to_operand(acc_S, pv_mma)
            warpgroup.fence()
            for kb in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
                cute.gemm(pv_mma, acc_O, tOrP[(None, None, kb)], tOrV[(None, None, kb, ks)], acc_O)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            if wiw == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(kv_empty + ks)
            ks = ks + 1
            if ks == 2:
                ks, kp = Int32(0), kp ^ 1
        acc_O_mn = acc_mn_view(acc_O)
        for r in cutlass.range_constexpr(nrow):
            rs = quad_reduce(row_sum[r], lambda x, y: x + y)
            inv = Float32(1.0) / rs
            if rs == 0.0:
                inv = Float32(1.0)
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv

        rO = cute.make_fragment_like(acc_O, BFloat16)
        rO.store(acc_O.load().to(BFloat16))
        sO = sQ[(None, None, cidx)]
        thr_O = cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16), pv_mma).get_slice(tid)
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16), thr_O.retile(rO), thr_O.partition_D(sO))
        self.wg_sync(cidx)
        thr_o = copy_O.get_slice(tid)
        tOsO = thr_o.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO)
        cute.autovec_copy(tOsO, tOrO)
        self.wg_sync(cidx)
        if wiw == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(q_empty)
        rps = 128 // (HD // 8)
        for m in cutlass.range_constexpr(cute.size(tOsO, mode=[1])):
            rq = j * QB + tid // (HD // 8) + rps * m
            if rq < Ls:
                cute.autovec_copy(tOrO[None, m, None],
                                  gvec(attn.iterator, (start + rq) * DQ + (G * g + cidx) * HD + (tid % (HD // 8)) * 8, BFloat16, 8))
        cute.arch.fence_proxy("async.global")
        self.wg_sync(cidx)
        if tid < 32:
            q0 = start + j * QB
            q1 = cutlass.min(q0 + QB, cu[s + 1]) - 1
            self.arrive_publish(self.fidx(SLOT_ATTN, l, q0 // BM), None, Int32(0), None, None, arrive, ready, expect)
            if q1 // BM != q0 // BM:
                self.arrive_publish(self.fidx(SLOT_ATTN, l, q1 // BM), None, Int32(0), None, None, arrive, ready, expect)
            self.arrive_publish(NSLOT * self.L1 * self.MB + l * self.Bmax + s, None, Int32(0), None, None, arrive, ready, expect)
        return ks, kp, qp

    @staticmethod
    def c2a_layout(c, a):
        return cute.make_layout((a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
                                stride=(c.stride[0], c.stride[1], (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2])))

    @cute.jit
    def acc_to_operand(self, acc, pv_mma):
        operand = cute.make_rmem_tensor_like(self.c2a_layout(acc.layout, pv_mma.tv_layout_A.shape[1]), BFloat16)
        cute.make_tensor(operand.iterator, acc.layout).store(acc.load().to(BFloat16))
        return operand