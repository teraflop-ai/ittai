import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, BFloat16, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

D, NH, NKV, HD, FF, L, VOCAB, EPS, THETA = 1024, 16, 8, 128, 3072, 28, 151669, 1e-6, 1e6
DQ, DKV = NH * HD, NKV * HD
DQKV, DGU = DQ + 2 * DKV, 2 * FF
K_OFF, V_OFF = DQ, DQ + DKV
G = NH // NKV
LOG2E = 1.4426950408889634
SCALE_LOG2 = HD ** -0.5 * LOG2E
OP_QKV, OP_OPROJ, OP_GU, OP_DOWN, OP_EMB, OP_ATTN, OP_FINAL = range(7)
NGEMM = 4
SLOT_X, SLOT_QKV, SLOT_ATTN, SLOT_XMID, SLOT_H, SLOT_SEQ = range(6)
NSLOT = 5
LN1, LN2 = 0, 1
TW = 8
BM, QB = 128, 64
SPEC = {
    OP_QKV: dict(N=DQKV, K=D, dep=SLOT_X, out=SLOT_QKV, dl=0, s_in=LN1, s_out=None, kind="qkv"),
    OP_OPROJ: dict(N=D, K=DQ, dep=SLOT_ATTN, out=SLOT_XMID, dl=0, s_in=None, s_out=LN2, kind="res"),
    OP_GU: dict(N=DGU, K=D, dep=SLOT_XMID, out=SLOT_H, dl=0, s_in=LN2, s_out=None, kind="gu"),
    OP_DOWN: dict(N=D, K=FF, dep=SLOT_H, out=SLOT_X, dl=1, s_in=None, s_out=LN1, kind="res"),
}


def acc_mn_view(acc):
    cm = cute.make_layout(acc.layout.shape)
    s0, st0 = cm.shape[0], cm.stride[0]
    if len(s0) == 2:
        shape = ((s0[1], cm.shape[1]), (s0[0], cm.shape[2]))
        stride = ((st0[1], cm.stride[1]), (st0[0], cm.stride[2]))
    else:
        shape = ((s0[1], cm.shape[1]), (s0[0], s0[2], cm.shape[2]))
        stride = ((st0[1], cm.stride[1]), (st0[0], st0[2], cm.stride[2]))
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, cute.make_layout(shape, stride=stride)))


def quad_reduce(val, op):
    val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
    return op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))


@cute.jit
def gptr(base, off, dtype: cutlass.Constexpr, align: cutlass.Constexpr):
    return cute.make_ptr(dtype, base.toint() + Int64(off) * (dtype.width // 8), cute.AddressSpace.gmem, assumed_align=align)


@cute.jit
def gvec(base, off, dtype: cutlass.Constexpr, n: cutlass.Constexpr):
    return cute.make_tensor(gptr(base, off, dtype, 16), cute.make_layout(n))


class MegaBase:

    def fidx(self, slot, l, i):
        return (slot * self.L1 + l) * self.MB + i

    @cute.jit
    def deps_bad(self, ready, d0: Int32, c0: Int32, d1: Int32, c1: Int32) -> Int32:
        bad = Int32(0)
        for k in cutlass.range(c0, unroll=1):
            if cute.arch.load(ready.iterator + (d0 + k), Int32, sem="acquire", scope="gpu") == 0:
                bad = bad + 1
        for k in cutlass.range(c1, unroll=1):
            if cute.arch.load(ready.iterator + (d1 + k), Int32, sem="acquire", scope="gpu") == 0:
                bad = bad + 1
        return bad

    @cute.jit
    def spin_ready(self, ready, d0: Int32, c0: Int32, d1: Int32, c1: Int32):
        bad = self.deps_bad(ready, d0, c0, d1, c1)
        while bad != 0:
            bad = self.deps_bad(ready, d0, c0, d1, c1)

    @cute.jit
    def arrive_publish(self, oidx: Int32, s_out: cutlass.Constexpr, row0: Int32, s2, stats, arrive, ready, expect):
        lane = cute.arch.lane_idx()
        old = Int32(0)
        if lane == 0:
            old = cute.arch.atomic_add(arrive.iterator + oidx, Int32(1), sem="acq_rel", scope="gpu")
        old = cute.arch.shuffle_sync(old, 0)
        if old == expect[oidx] - 1:
            if const_expr(s_out is not None):
                cute.arch.load(arrive.iterator + oidx, Int32, sem="acquire", scope="gpu")
                for rr in cutlass.range_constexpr(BM // 32):
                    frow = row0 + lane + 32 * rr
                    b = Float32(0.0)
                    for tt in cutlass.range_constexpr(self.n_part):
                        b = b + s2[s_out, frow, tt]
                    stats[s_out, frow] = cute.math.rsqrt(b * Float32(1.0 / D) + Float32(EPS), fastmath=True)
                cute.arch.fence_acq_rel_gpu()
                cute.arch.sync_warp()
            if lane == 0:
                cute.arch.store(ready.iterator + oidx, Int32(1), sem="release", scope="gpu")

    @cute.jit
    def epi_chunk(self, kind: cutlass.Constexpr, s_out: cutlass.Constexpr, l: Int32, j: Int32, row: Int32, col: Int32,
                  lane: Int32, v8, h, qkv, act, s2, qk_w, rope, pos):
        x = [Float32(v8[e]) for e in range(8)]
        if const_expr(kind == "qkv"):
            if col < V_OFF:
                xb = [Float32(BFloat16(v)) for v in x]
                ss = xb[0] * xb[0]
                for e in cutlass.range_constexpr(1, 8):
                    ss = ss + xb[e] * xb[e]
                for off in (8, 4, 2, 1):
                    ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=off, mask=-1, mask_and_clamp=31)
                rq = cute.math.rsqrt(ss * Float32(1.0 / HD) + Float32(EPS), fastmath=True)
                is_k = Int32(0)
                if col >= K_OFF:
                    is_k = Int32(1)
                c_in = (lane * 8) % HD
                gam = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                rp = cute.make_rmem_tensor(cute.make_layout(8), Float32)
                cute.autovec_copy(gvec(qk_w.iterator, (l * 2 + is_k) * HD + c_in, Float32, 8), gam)
                cute.autovec_copy(gvec(rope.iterator, pos[row] * HD + c_in, Float32, 8), rp)
                o = cute.make_rmem_tensor(cute.make_layout(8), BFloat16)
                for p in cutlass.range_constexpr(4):
                    y0, y1 = xb[2 * p] * rq * gam[2 * p], xb[2 * p + 1] * rq * gam[2 * p + 1]
                    cs, sn = rp[2 * p], rp[2 * p + 1]
                    o[2 * p] = BFloat16(y0 * cs - y1 * sn)
                    o[2 * p + 1] = BFloat16(y1 * cs + y0 * sn)
                cute.autovec_copy(o, gvec(qkv.iterator, row * DQKV + col, BFloat16, 8))
            else:
                cute.autovec_copy(v8, gvec(qkv.iterator, row * DQKV + col, BFloat16, 8))
        elif const_expr(kind == "res"):
            r = cute.make_rmem_tensor(cute.make_layout(8), BFloat16)
            cute.autovec_copy(gvec(h.iterator, row * D + col, BFloat16, 8), r)
            o = cute.make_rmem_tensor(cute.make_layout(8), BFloat16)
            ss = Float32(0.0)
            for e in cutlass.range_constexpr(8):
                y = BFloat16(x[e] + Float32(r[e]))
                o[e] = y
                ss = ss + Float32(y) * Float32(y)
            cute.autovec_copy(o, gvec(h.iterator, row * D + col, BFloat16, 8))
            for off in self.red_offs:
                ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=off, mask=-1, mask_and_clamp=31)
            if lane == 0:
                s2[s_out, row, j] = ss
        else:
            o = cute.make_rmem_tensor(cute.make_layout(4), BFloat16)
            for p in cutlass.range_constexpr(4):
                g, u = x[2 * p], x[2 * p + 1]
                o[p] = BFloat16(g * u / (Float32(1.0) + cute.math.exp2(-g * Float32(LOG2E), fastmath=True)))
            cute.autovec_copy(o, cute.make_tensor(gptr(act.iterator, row * FF + col // 2, BFloat16, 8), cute.make_layout(4)))

    @cute.jit
    def final_pool(self, lane: Int32, s: Int32, cu, h, stats, fin_w, out):
        r = cu[s + 1] - 1
        rstd = stats[LN1, r]
        xf = cute.make_rmem_tensor(cute.make_layout(32), BFloat16)
        wf = cute.make_rmem_tensor(cute.make_layout(32), Float32)
        cute.autovec_copy(gvec(h.iterator, r * D + lane * 32, BFloat16, 32), xf)
        cute.autovec_copy(gvec(fin_w.iterator, lane * 32, Float32, 32), wf)
        y = xf.load().to(Float32) * rstd * wf.load()
        ss = (y * y).reduce(cute.ReductionOp.ADD, Float32.zero, 0)
        for i in cutlass.range_constexpr(5):
            ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=1 << i, mask=-1, mask_and_clamp=31)
        yf = cute.make_rmem_tensor(cute.make_layout(32), Float32)
        yf.store(y * cute.math.rsqrt(ss, fastmath=True))
        cute.autovec_copy(yf, gvec(out.iterator, s * D + lane * 32, Float32, 32))

    @cute.jit
    def embed_row(self, r: Int32, T: Int32, embed, tokens, h, stats):
        ss = Float32(0.0)
        tok = Int32(0)
        if r < T:
            tok = tokens[r]
        for ch in cutlass.range_constexpr(D // 64):
            fr = cute.make_rmem_tensor(cute.make_layout(64), BFloat16)
            if r < T:
                cute.autovec_copy(gvec(embed.iterator, tok * D + ch * 64, BFloat16, 64), fr)
            else:
                fr.fill(0.0)
            cute.autovec_copy(fr, gvec(h.iterator, r * D + ch * 64, BFloat16, 64))
            x = fr.load().to(Float32)
            ss = ss + (x * x).reduce(cute.ReductionOp.ADD, Float32.zero, 0)
        stats[LN1, r] = cute.math.rsqrt(ss * Float32(1.0 / D) + Float32(EPS), fastmath=True)

    def _c_layout(self, tn, rows=BM):
        atom = cute.make_composed_layout(cute.make_swizzle(3, 3, int(math.log2(tn)) - 3), 0,
                                         cute.make_layout((8, tn), stride=(tn, 1)))
        return cute.tile_to_shape(atom, (rows, tn), (0, 1))

    def _copy_rows(self, atom, cols, threads):
        lanes = cols // 8
        return cute.make_tiled_copy_tv(atom, cute.make_layout((threads // lanes, lanes), stride=(lanes, 1)), cute.make_layout((1, 8)))


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
        A, W, Dd = [h, attn, h, act], [wqkv, wo, wgu, wd], [qkv, h, act, h]
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


def rope_perm(n_heads):
    return np.concatenate([[hh * HD + i, hh * HD + i + HD // 2] for hh in range(n_heads) for i in range(HD // 2)])


def build_tasks(seqlens, MB_alloc, Bmax, L1, n_tiles, spt, kvb, group_m):
    seqlens = [int(x) for x in seqlens]
    B = len(seqlens)
    cu = np.concatenate([[0], np.cumsum(seqlens)])
    T = int(cu[-1])
    MB = (T + BM - 1) // BM
    n_qkv, n_o, n_gu, n_d = n_tiles
    seq_of_row = np.repeat(np.arange(B), seqlens)
    s_lo = np.array([seq_of_row[m * BM] for m in range(MB)])
    s_hi = np.array([seq_of_row[min((m + 1) * BM, T) - 1] for m in range(MB)])
    fidx = lambda slot, l, i: (slot * L1 + l) * MB_alloc + i
    seq_base = NSLOT * L1 * MB_alloc
    NF = seq_base + L * Bmax
    expect = np.zeros(NF, dtype=np.int32)

    at = []
    for s in range(B):
        for b in reversed(range((seqlens[s] + QB - 1) // QB)):
            q0 = int(cu[s]) + b * QB
            q1 = min(q0 + QB, int(cu[s + 1])) - 1
            hi = min(q0 + QB - 1, int(cu[s + 1]) - 1) // BM
            for g in range(NKV):
                at.append((s, g, b, q0 // BM, q1 // BM, int(cu[s]) // BM, hi))
    at = np.array(at, dtype=np.int64).reshape(-1, 7)
    mbs = np.arange(MB)
    for l in range(L):
        expect[fidx(SLOT_QKV, l, mbs)] = spt * n_qkv
        expect[fidx(SLOT_XMID, l, mbs)] = spt * n_o
        expect[fidx(SLOT_H, l, mbs)] = spt * n_gu
        expect[fidx(SLOT_X, l + 1, mbs)] = spt * n_d
        np.add.at(expect, fidx(SLOT_ATTN, l, at[:, 3]), spt)
        straddle = at[:, 4] != at[:, 3]
        np.add.at(expect, fidx(SLOT_ATTN, l, at[straddle, 4]), spt)
        np.add.at(expect, seq_base + l * Bmax + at[:, 0], spt)
    expect[fidx(SLOT_X, 0, mbs)] = spt

    def gemm(op, nn, dep):
        rows = []
        for gm in range(0, MB, group_m):
            ii = np.arange(gm, min(gm + group_m, MB))
            jj = np.arange(nn)
            i_, j_ = np.meshgrid(ii, jj)
            i_, j_ = i_.ravel(), j_.ravel()
            r = np.zeros((len(i_), TW), dtype=np.int64)
            r[:, 0], r[:, 2], r[:, 3], r[:, 4], r[:, 5], r[:, 6] = op, i_, j_, fidx(dep, 0, i_), 1, -1
            rows.append(r)
        return np.concatenate(rows)

    g_qkv, g_o, g_gu, g_d = gemm(OP_QKV, n_qkv, SLOT_X), gemm(OP_OPROJ, n_o, SLOT_ATTN), gemm(OP_GU, n_gu, SLOT_XMID), gemm(OP_DOWN, n_d, SLOT_H)
    a0 = np.zeros((len(at), TW), dtype=np.int64)
    a0[:, 0], a0[:, 2], a0[:, 3] = OP_ATTN, at[:, 0] * NKV + at[:, 1], at[:, 2]
    a0[:, 4], a0[:, 5], a0[:, 6] = fidx(SLOT_QKV, 0, at[:, 5]), at[:, 6] - at[:, 5] + 1, -1
    emb = np.zeros((MB, TW), dtype=np.int64)
    emb[:, 0], emb[:, 2], emb[:, 4], emb[:, 6] = OP_EMB, mbs, -1, -1
    parts = [emb]
    for l in range(L):
        for g0 in (g_qkv, a0, g_o, g_gu, g_d):
            r = g0.copy()
            r[:, 1] = l
            r[:, 4] += l * MB_alloc
            if g0 is g_qkv and l > 0:
                r[:, 6] = seq_base + (l - 1) * Bmax + s_lo[r[:, 2]]
                r[:, 7] = s_hi[r[:, 2]] - s_lo[r[:, 2]] + 1
            parts.append(r)
    fin = np.zeros((B, TW), dtype=np.int64)
    fin[:, 0], fin[:, 2], fin[:, 4], fin[:, 5], fin[:, 6] = OP_FINAL, np.arange(B), fidx(SLOT_X, L, (cu[1:] - 1) // BM), 1, -1
    parts.append(fin)
    return np.concatenate(parts).astype(np.int32), expect, T


class Engine:
    @torch.no_grad()
    def __init__(self, sd, max_tokens, max_seqs, arch, tile_n, stages, kv_block, group_m, warps=8, device="cuda", graph=True):
        dev = torch.device(device)
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        M = (max_tokens + BM - 1) // BM * BM
        self.M, self.Bmax, self.MB, self.L1, self.dev, self.group_m = M, max_seqs, M // BM, L + 1, dev, group_m
        self.arch = arch
        self.kern = (MegaSm90(tile_n, stages, kv_block, M, max_seqs) if arch == "sm90"
                     else MegaSm80(tile_n, stages, warps, kv_block, M, max_seqs))
        self.spt = 2 if arch == "sm90" else 1
        self.n_tiles = (DQKV // tile_n, D // tile_n, DGU // tile_n, D // tile_n)
        self.kvb = kv_block
        f32 = lambda t: t.detach().float()
        bf = lambda t: t.to(torch.bfloat16).contiguous().to(dev)
        pq, pk = torch.from_numpy(rope_perm(NH)), torch.from_numpy(rope_perm(NKV))
        Wqkv, Wo, Wgu, Wd, qkn = [], [], [], [], []
        for l in range(L):
            p = f"layers.{l}."
            g1, g2 = f32(sd[p + "input_layernorm.weight"]), f32(sd[p + "post_attention_layernorm.weight"])
            q, k, v = (f32(sd[p + f"self_attn.{n}_proj.weight"]) * g1[None] for n in "qkv")
            Wqkv.append(torch.cat([q[pq], k[pk], v]))
            Wo.append(f32(sd[p + "self_attn.o_proj.weight"]))
            gg, uu = (f32(sd[p + f"mlp.{n}_proj.weight"]) * g2[None] for n in ("gate", "up"))
            Wgu.append(torch.stack([gg, uu], 1).reshape(DGU, D))
            Wd.append(f32(sd[p + "mlp.down_proj.weight"]))
            qkn.append(torch.stack([f32(sd[p + "self_attn.q_norm.weight"])[pq[:HD]], f32(sd[p + "self_attn.k_norm.weight"])[pk[:HD]]]))
        self.w = [bf(torch.cat(Wqkv)), bf(torch.cat(Wo)), bf(torch.cat(Wgu)), bf(torch.cat(Wd))]
        self.embed = bf(sd["embed_tokens.weight"])
        self.qk_w = torch.stack(qkn).contiguous().to(dev)
        self.fin_w = f32(sd["norm.weight"]).contiguous().to(dev)
        inv = 1.0 / (THETA ** (np.arange(0, HD, 2) / HD))
        ang = np.arange(M)[:, None] * inv[None]
        rope = np.empty((M, HD), np.float32)
        rope[:, 0::2], rope[:, 1::2] = np.cos(ang), np.sin(ang)
        self.rope = torch.from_numpy(rope).to(dev)
        z = lambda *s, dt=torch.bfloat16: torch.zeros(*s, dtype=dt, device=dev)
        self.h, self.qkv, self.attn, self.act = z(M, D), z(M + 128, DQKV), z(M, DQ), z(M, FF)
        self.stats, self.s2 = z(2, M, dt=torch.float32), z(2, M, D // tile_n, dt=torch.float32)
        self.tokens, self.pos = z(M, dt=torch.int32), z(M, dt=torch.int32)
        self.cu = z(max_seqs + 1, dt=torch.int32)
        self.out = z(max_seqs, D, dt=torch.float32)
        NF = NSLOT * self.L1 * self.MB + L * max_seqs
        nt_max = self.MB * (sum(self.n_tiles) * L + 1) + (self.MB * 2 + max_seqs) * NKV * L + max_seqs
        self.tasks = z(nt_max, TW, dt=torch.int32)
        self.expect, self.arrive = z(NF, dt=torch.int32), z(NF, dt=torch.int32)
        self.ready = z(NF, dt=torch.int32)
        self.ctr, self.meta = z(1, dt=torch.int32), z(4, dt=torch.int32)
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.grid = sms if arch == "sm90" else sms * (2 if (warps == 4 and torch.cuda.get_device_capability(dev) == (8, 0)) else 1)
        self.torch_args = [self.h, self.qkv, self.attn, self.act, *self.w, self.embed, self.tokens, self.pos, self.cu, self.stats,
                           self.s2, self.qk_w, self.rope, self.fin_w, self.out, self.tasks, self.expect, self.arrive, self.ready,
                           self.ctr, self.meta]
        self.cute_args = [from_dlpack(t, assumed_align=16) for t in self.torch_args]
        self.compiled = cute.compile(self.kern, *self.cute_args, Int32(self.grid), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        self.graph, self._cache = (None if graph else False), {}

    def _tasks(self, seqlens):
        key = tuple(seqlens)
        if key not in self._cache:
            tasks, expect, T = build_tasks(seqlens, self.MB, self.Bmax, self.L1, self.n_tiles, self.spt, self.kvb, self.group_m)
            dev = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)).to(self.dev)
            self._cache[key] = (dev(tasks), dev(expect), dev(np.concatenate([np.arange(n) for n in seqlens])),
                                dev(np.concatenate([[0], np.cumsum(seqlens)])), dev([len(tasks), T, len(seqlens), 0]))
        return self._cache[key]

    def _graph_body(self):
        self.arrive.zero_(); self.ready.zero_(); self.ctr.zero_()
        self.compiled(*self.cute_args, Int32(self.grid), cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    @torch.no_grad()
    def forward(self, input_ids):
        seqlens = [len(x) for x in input_ids]
        T, B = sum(seqlens), len(seqlens)
        assert T <= self.M and B <= self.Bmax
        tasks, expect, pos, cu, meta = self._tasks(seqlens)
        self.tasks[: len(tasks)].copy_(tasks)
        self.expect.copy_(expect)
        self.pos[:T].copy_(pos)
        self.cu[: B + 1].copy_(cu)
        self.meta.copy_(meta)
        self.tokens[:T].copy_(torch.tensor([t for s in input_ids for t in s], dtype=torch.int32))
        if self.graph is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._graph_body()
            torch.cuda.current_stream().wait_stream(s)
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._graph_body()
                self.graph = g
            except Exception as ex:
                print(f"CUDA graph capture failed ({ex}); launching eagerly")
                self.graph = False
        if self.graph:
            self.graph.replay()
        else:
            self._graph_body()
        return self.out[:B]


class Qwen3Embedder:
    CONFIGS = {
        "sm90": [dict(tile_n=t, stages=s, kv_block=k, group_m=g) for t, s in ((256, 3), (128, 4), (128, 5)) for k in (64, 128) for g in (8, 16)],
        "sm80": [dict(warps=w, tile_n=t, stages=s, kv_block=k, group_m=g)
                 for w, t, s in ((4, 128, 4), (8, 128, 4), (8, 128, 5), (8, 256, 3), (8, 256, 4))
                 for k in (64, 128) for g in (8, 16) if w == 8 or k == 64],
    }
    CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_mega_tune.json")

    def __init__(self, model_id="Qwen/Qwen3-Embedding-0.6B", max_tokens=8192, max_seqs=256, device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device).eval()
        self.sd = self.model.state_dict()
        self.max_tokens, self.max_seqs, self.device, self.model_id = max_tokens, max_seqs, device, model_id
        cc = torch.cuda.get_device_capability()
        self.arch = "sm90" if cc[0] == 9 else "sm80"
        self.smem = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
        self.engine = Engine(self.sd, max_tokens, max_seqs, self.arch, **self._tune(model_id))

    def _fits(self, cfg):
        k = MegaSm90(cfg["tile_n"], cfg["stages"], cfg["kv_block"], BM, 1) if self.arch == "sm90" else \
            MegaSm80(cfg["tile_n"], cfg["stages"], cfg["warps"], cfg["kv_block"], BM, 1)
        return k.smem_bytes + 1024 <= self.smem

    def _ref(self, texts):
        batch = self.tok(texts, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hs = self.model(**batch).last_hidden_state
        ids = [[t for t, m in zip(i, mm) if m] for i, mm in zip(batch["input_ids"].tolist(), batch["attention_mask"].tolist())]
        return ids, F.normalize(hs[:, -1].float(), dim=-1)

    def _time(self, engine, ids, iters=10):
        for _ in range(3):
            engine.forward(ids)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            engine.forward(ids)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    def _tune(self, model_id):
        key = f"{torch.cuda.get_device_name()}|{model_id}|{self.max_tokens}"
        cache = json.load(open(self.CACHE)) if os.path.exists(self.CACHE) else {}
        if key not in cache:
            texts = ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of China?",
                     "The capital of China is Beijing.", "Gravity is a fundamental force. " * 40, "Explain gravity"] * 8
            ids, ref = self._ref(texts)
            timed = []
            for c in self.CONFIGS[self.arch]:
                if not self._fits(c):
                    continue
                try:
                    engine = Engine(self.sd, self.max_tokens, self.max_seqs, self.arch, **c)
                    out = engine.forward(ids).float()
                    cos = F.cosine_similarity(out, ref, dim=-1).min().item()
                    ms = self._time(engine, ids) if cos > 0.999 else float("nan")
                    if cos > 0.999:
                        timed.append((ms, c))
                    print(f"{c}: cos={cos:.5f} {'ok' if cos > 0.999 else 'MISMATCH'} {ms:.2f} ms")
                    del engine
                except Exception as ex:
                    print(f"{c}: failed ({type(ex).__name__}: {ex})")
                torch.cuda.empty_cache()
            if not timed:
                raise RuntimeError("no kernel config passed validation (see messages above)")
            cache[key] = min(timed, key=lambda t: t[0])[1]
            json.dump(cache, open(self.CACHE, "w"), indent=1)
        return cache[key]

    @torch.no_grad()
    def embed(self, texts, max_length=2048):
        ids = [self.tok(t, truncation=True, max_length=max_length)["input_ids"] for t in texts]
        out, k = [], 0
        while k < len(ids):
            n, tot = 0, 0
            while k + n < len(ids) and n < self.max_seqs and tot + len(ids[k + n]) <= self.max_tokens:
                tot += len(ids[k + n]); n += 1
            assert n > 0, "sequence longer than max_tokens"
            out.append(self.engine.forward(ids[k: k + n]).clone())
            k += n
        return torch.cat(out)