import json
import math
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, Float16, const_expr
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

OP_EMB, OP_QKV, OP_OPROJ, OP_FC1, OP_FC2, OP_ATTN = range(6)
NGEMM = 5
SLOT_X, SLOT_QKV, SLOT_ATTN, SLOT_XMID, SLOT_H = range(5)
NSLOT = 5
LN1, LN2 = 0, 1


def acc_mn_view(acc):
    cm = cute.make_layout(acc.layout.shape)
    shape = ((cm.shape[0][1], cm.shape[1]), (cm.shape[0][0], cm.shape[2]))
    stride = ((cm.stride[0][1], cm.stride[1]), (cm.stride[0][0], cm.stride[2]))
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, cute.make_layout(shape, stride=stride)))


def quad_reduce(val, op):
    val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
    return op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))


class MegaSm86:
    def __init__(self, D, Fd, S, H, hd, hd0, L, n_i, tile_n, stages, warps, fold_k, kv_block, sep_c):
        self.bM, self.bN, self.bK = 128, tile_n, 32
        self.cta_tiler = (self.bM, self.bN, self.bK)
        self.num_stages, self.fold_k = stages, fold_k
        self.atom_layout_mnk = (2, warps // 2, 1)
        self.num_warps = warps
        self.num_threads = warps * 32
        self.dtype = Float16
        self.D, self.S, self.H, self.L1 = D, S, H, L + 1
        self.hd, self.nq, self.kv_block = hd, S // self.bM, kv_block
        self.n_i, self.sep_c = n_i, sep_c
        tn = tile_n
        self.spec = {
            OP_EMB: dict(dep=None, out=SLOT_X, dl=0, expect=D // tn, by_img=False, aux=True, s_in=None, s_out=LN1, gelu=False),
            OP_QKV: dict(dep=SLOT_X, out=SLOT_QKV, dl=0, expect=self.nq * (3 * H * hd // tn), by_img=True, aux=False, s_in=LN1, s_out=None, gelu=False),
            OP_OPROJ: dict(dep=SLOT_ATTN, out=SLOT_XMID, dl=0, expect=D // tn, by_img=False, aux=True, s_in=None, s_out=LN2, gelu=False),
            OP_FC1: dict(dep=SLOT_XMID, out=SLOT_H, dl=0, expect=Fd // tn, by_img=False, aux=False, s_in=LN2, s_out=None, gelu=True),
            OP_FC2: dict(dep=SLOT_H, out=SLOT_X, dl=1, expect=D // tn, by_img=False, aux=True, s_in=None, s_out=LN1, gelu=False),
        }
        self.softmax_scale_log2 = (hd0 ** -0.5) * 1.4426950408889634

    def fidx(self, slot, l, i):
        return (slot * self.L1 + l) * self.n_i + i

    @cute.jit
    def __call__(self, a0, a1, a2, a3, a4, w0, w1, w2, w3, w4, d0, d1, d2, d3, d4, x0, x1, x2, x3, x4,
                 b0, b1, b2, b3, b4, c0, c1, c2, c3, c4, stats, s1, s2, key_len, tasks, task_ctr, arrive, ready,
                 task_begin: Int32, task_end: Int32, eps: Float32, grid: Int32, stream: cuda.CUstream):
        dt = self.dtype
        sA_layout, sA_swizzle = self._make_smem_layout_AB(dt, 128, (self.bM, self.bK, self.num_stages))
        sB_layout, sB_swizzle = self._make_smem_layout_AB(dt, 128, (self.bN, self.bK, self.num_stages))
        sC_layout = self._make_smem_layout_C(dt, 128, (self.bM, self.bN))
        atom_async = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL), dt,
                                         num_bits_per_copy=128)
        atom_sync = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dt, num_bits_per_copy=128)
        tiled_copy_AB = self._make_gmem_tiled_copy(atom_async, dt, self.bK, 128)
        tiled_copy_C = self._make_gmem_tiled_copy(atom_sync, dt, self.bN, 128)
        al = self.atom_layout_mnk
        tiled_mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(dt, Float16, (16, 8, 16)), cute.make_layout(al),
                                        permutation_mnk=(al[0] * 16, al[1] * 16, al[2] * 16))
        sQ_layout = cute.make_layout((self.bM, self.hd), stride=(self.hd + 8, 1))
        sKV_layout = cute.make_layout((self.kv_block, self.hd), stride=(self.hd + 8, 1))
        thr_qkv = cute.make_layout((self.kv_block, 2), stride=(2, 1))
        tiled_copy_QKV = cute.make_tiled_copy_tv(atom_async, thr_qkv, cute.make_layout((1, self.hd // 2)))
        tiled_copy_O = cute.make_tiled_copy_tv(atom_sync, thr_qkv, cute.make_layout((1, self.hd // 2)))
        tiled_mma_attn = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(dt, Float32, (16, 8, 16)),
                                             cute.make_layout((self.num_warps, 1, 1)), permutation_mnk=(self.num_warps * 16, 16, 16))
        tiled_mma_pv = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(dt, Float16, (16, 8, 16)),
                                           cute.make_layout((self.num_warps, 1, 1)), permutation_mnk=(self.num_warps * 16, 16, 16))
        self.kernel(a0, a1, a2, a3, a4, w0, w1, w2, w3, w4, d0, d1, d2, d3, d4, x0, x1, x2, x3, x4,
                    b0, b1, b2, b3, b4, c0, c1, c2, c3, c4, stats, s1, s2, key_len, tasks, task_ctr, arrive, ready,
                    task_begin, task_end, eps, sA_layout, sA_swizzle, sB_layout, sB_swizzle, sC_layout, sQ_layout, sKV_layout,
                    tiled_copy_AB, tiled_copy_C, tiled_mma, tiled_copy_QKV, tiled_copy_O, tiled_mma_attn, tiled_mma_pv,
                    ).launch(grid=[grid, 1, 1], block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, a0, a1, a2, a3, a4, w0, w1, w2, w3, w4, d0, d1, d2, d3, d4, x0, x1, x2, x3, x4,
               b0, b1, b2, b3, b4, c0, c1, c2, c3, c4, stats, s1, s2, key_len, tasks, task_ctr, arrive, ready,
               task_begin: Int32, task_end: Int32, eps: Float32,
               sA_layout, sA_swizzle, sB_layout, sB_swizzle, sC_layout, sQ_layout, sKV_layout,
               tiled_copy_AB, tiled_copy_C, tiled_mma, tiled_copy_QKV, tiled_copy_O, tiled_mma_attn, tiled_mma_pv):
        A, W, Dd = [a0, a1, a2, a3, a4], [w0, w1, w2, w3, w4], [d0, d1, d2, d3, d4]
        AUX, BIAS, CVEC = [x0, x1, x2, x3, x4], [b0, b1, b2, b3, b4], [c0, c1, c2, c3, c4]
        tidx, _, _ = cute.arch.thread_idx()
        dt = self.dtype

        @cute.struct
        class SmemAB:
            a: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sA_layout)], 1024]
            b: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sB_layout)], 1024]

        @cute.struct
        class SmemC:
            c: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sC_layout)], 1024]

        @cute.struct
        class SmemAttn:
            q: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_layout)], 1024]
            k: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sKV_layout)], 1024]
            v: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sKV_layout)], 1024]

        smem = SmemAllocator()
        storage = smem.allocate(max(SmemAB.size_in_bytes(), SmemAttn.size_in_bytes()), byte_alignment=1024)
        if const_expr(self.sep_c):
            storage_c = smem.allocate(SmemC.size_in_bytes(), byte_alignment=1024)
        else:
            storage_c = storage
        s_task = smem.allocate_tensor(Int32, 8, byte_alignment=16)
        sA = SmemAB(storage).a.get_tensor(sA_layout, swizzle=sA_swizzle)
        sB = SmemAB(storage).b.get_tensor(sB_layout, swizzle=sB_swizzle)
        sC = SmemC(storage_c).c.get_tensor(sC_layout)
        sQ = SmemAttn(storage).q.get_tensor(sQ_layout)
        sK = SmemAttn(storage).k.get_tensor(sKV_layout)
        sV = SmemAttn(storage).v.get_tensor(sKV_layout)

        if tidx == 0:
            s_task[0] = task_begin + cute.arch.atomic_add(task_ctr.iterator, Int32(1), sem="relaxed", scope="gpu")
            s_task[1] = task_begin + cute.arch.atomic_add(task_ctr.iterator, Int32(1), sem="relaxed", scope="gpu")
        cute.arch.sync_threads()
        t, tn = s_task[0], s_task[1]
        cute.arch.sync_threads()
        tc = cutlass.min(t, task_end - 1)
        op, l, i, j = tasks[tc, 0], tasks[tc, 1], tasks[tc, 2], tasks[tc, 3]
        pre = Int32(0)
        while t < task_end:
            for op_c in cutlass.range_constexpr(NGEMM):
                if op == op_c:
                    if pre == 0:
                        if const_expr(self.spec[op_c]["dep"] is not None):
                            self.wait_ready(ready, self.fidx(self.spec[op_c]["dep"], l, i), tidx)
                        self.gemm_prologue(op_c, l, i, j, A[op_c], W[op_c], sA, sB, tiled_copy_AB, tidx)
                    opn, ln, in_, jn, fl, tn2 = Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
                    if tidx == 0:
                        tn2 = task_begin + cute.arch.atomic_add(task_ctr.iterator, Int32(1), sem="relaxed", scope="gpu")
                        tcn = cutlass.min(tn, task_end - 1)
                        opn, ln, in_, jn = tasks[tcn, 0], tasks[tcn, 1], tasks[tcn, 2], tasks[tcn, 3]
                        fl = cute.arch.load(ready.iterator + tasks[tcn, 4], Int32, sem="acquire", scope="gpu")
                    acc = self.gemm_mainloop(op_c, l, i, j, A[op_c], W[op_c], sA, sB, tiled_copy_AB, tiled_mma, tidx)
                    if tidx == 0:
                        s_task[0], s_task[1], s_task[2], s_task[3], s_task[5] = opn, ln, in_, jn, tn2
                        s_task[4] = fl if (tn < task_end and opn < NGEMM and const_expr(self.sep_c)) else Int32(0)
                    cute.arch.sync_threads()
                    opn, ln, in_, jn, pre, tn2 = s_task[0], s_task[1], s_task[2], s_task[3], s_task[4], s_task[5]
                    if pre != 0:
                        for op_nc in cutlass.range_constexpr(NGEMM):
                            if opn == op_nc:
                                self.gemm_prologue(op_nc, ln, in_, jn, A[op_nc], W[op_nc], sA, sB, tiled_copy_AB, tidx)
                    self.gemm_epilogue(op_c, l, i, j, acc, Dd[op_c], AUX[op_c], BIAS[op_c], CVEC[op_c],
                                       stats, s1, s2, arrive, ready, eps, sC, tiled_copy_C, tiled_mma, tidx)
                    t, tn = tn, tn2
                    op, l, i, j = opn, ln, in_, jn
            if op == OP_ATTN:
                tn2 = Int32(0)
                if tidx == 0:
                    tn2 = task_begin + cute.arch.atomic_add(task_ctr.iterator, Int32(1), sem="relaxed", scope="gpu")
                self.run_attn(l, i, j, d1, a2, key_len, arrive, ready, sQ, sK, sV, sQ_layout,
                              tiled_copy_QKV, tiled_copy_O, tiled_mma_attn, tiled_mma_pv, tidx)
                if tidx == 0:
                    s_task[5] = tn2
                cute.arch.sync_threads()
                t, tn = tn, s_task[5]
                cute.arch.sync_threads()
                tc = cutlass.min(t, task_end - 1)
                op, l, i, j = tasks[tc, 0], tasks[tc, 1], tasks[tc, 2], tasks[tc, 3]
                pre = Int32(0)

    @cute.jit
    def wait_ready(self, ready, idx: Int32, tidx: Int32):
        if tidx == 0:
            v = cute.arch.load(ready.iterator + idx, Int32, sem="acquire", scope="gpu")
            while v == 0:
                v = cute.arch.load(ready.iterator + idx, Int32, sem="acquire", scope="gpu")
        cute.arch.sync_threads()

    def _ab_partitions(self, mA, mW, l, i, j, sA, sB, tiled_copy_AB, tidx):
        lb = l if const_expr(mW.shape[0] > 1) else Int32(0)
        coord = (i, j, None)
        gA = cute.local_tile(mA, tiler=self.cta_tiler, coord=coord, proj=(1, None, 1))
        gB = cute.local_tile(mW[lb, None, None], tiler=self.cta_tiler, coord=coord, proj=(None, 1, 1))
        thr_copy = tiled_copy_AB.get_slice(tidx)
        return thr_copy.partition_S(gA), thr_copy.partition_D(sA), thr_copy.partition_S(gB), thr_copy.partition_D(sB)

    @cute.jit
    def gemm_prologue(self, op: cutlass.Constexpr[int], l: Int32, i: Int32, j: Int32, mA, mW, sA, sB, tiled_copy_AB, tidx: Int32):
        tAgA, tAsA, tBgB, tBsB = self._ab_partitions(mA, mW, l, i, j, sA, sB, tiled_copy_AB, tidx)
        for s in cutlass.range_constexpr(self.num_stages - 1):
            cute.copy(tiled_copy_AB, tAgA[None, None, None, s], tAsA[None, None, None, s])
            cute.copy(tiled_copy_AB, tBgB[None, None, None, s], tBsB[None, None, None, s])
            cute.arch.cp_async_commit_group()

    @cute.jit
    def gemm_mainloop(self, op: cutlass.Constexpr[int], l: Int32, i: Int32, j: Int32, mA, mW, sA, sB, tiled_copy_AB,
                      tiled_mma, tidx: Int32):
        tAgA, tAsA, tBgB, tBsB = self._ab_partitions(mA, mW, l, i, j, sA, sB, tiled_copy_AB, tidx)
        num_smem_stages = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        k_tile_index = Int32(num_smem_stages - 1)
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA, tCsB = thr_mma.partition_A(sA), thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrH = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.bM, self.bN)), Float16)
        tCrC = cute.make_fragment_like(tCrH, Float32)
        tCrH.fill(0.0)
        tCrC.fill(0.0)
        atom_s2r = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), self.dtype)
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r, tiled_mma)
        thr_s2r_A, thr_s2r_B = tiled_s2r_A.get_slice(tidx), tiled_s2r_B.get_slice(tidx)
        tCsA_cv, tCrA_cv = thr_s2r_A.partition_S(sA), thr_s2r_A.retile(tCrA)
        tCsB_cv, tCrB_cv = thr_s2r_B.partition_S(sB), thr_s2r_B.retile(tCrB)
        smem_pipe_read = Int32(0)
        smem_pipe_write = Int32(num_smem_stages - 1)
        tCsA_p = tCsA_cv[None, None, None, smem_pipe_read]
        tCsB_p = tCsB_cv[None, None, None, smem_pipe_read]
        num_k_block = cute.size(tCrA, mode=[2])
        if num_k_block > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(tiled_s2r_A, tCsA_p[None, None, 0], tCrA_cv[None, None, 0])
            cute.copy(tiled_s2r_B, tCsB_p[None, None, 0], tCrB_cv[None, None, 0])
        for k_tile in range(k_tile_count):
            for k_block in cutlass.range(num_k_block, unroll_full=True):
                if k_block == num_k_block - 1:
                    tCsA_p = tCsA_cv[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB_cv[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()
                k_block_next = (k_block + 1) % num_k_block
                cute.copy(tiled_s2r_A, tCsA_p[None, None, k_block_next], tCrA_cv[None, None, k_block_next])
                cute.copy(tiled_s2r_B, tCsB_p[None, None, k_block_next], tCrB_cv[None, None, k_block_next])
                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        cute.copy(tiled_copy_AB, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, smem_pipe_write])
                        cute.copy(tiled_copy_AB, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, smem_pipe_write])
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == num_smem_stages:
                        smem_pipe_read = Int32(0)
                cute.gemm(tiled_mma, tCrH, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrH)
            if (k_tile + 1) % self.fold_k == 0:
                tCrC.store(tCrC.load() + tCrH.load().to(Float32))
                tCrH.fill(0.0)
        if const_expr(k_tile_count % self.fold_k != 0):
            tCrC.store(tCrC.load() + tCrH.load().to(Float32))
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        return tCrC

    @cute.jit
    def gemm_epilogue(self, op: cutlass.Constexpr[int], l: Int32, i: Int32, j: Int32, tCrC, mD, mAux, mBias, mC,
                      stats, s1, s2, arrive, ready, eps: Float32, sC, tiled_copy_C, tiled_mma, tidx: Int32):
        spec = self.spec[op]
        Dm = mD.shape[1]
        row0, col0 = i * self.bM, j * self.bN
        lb = l if const_expr(mBias.shape[0] > 1) else Int32(0)
        thr_mma = tiled_mma.get_slice(tidx)
        gD = cute.local_tile(mD, tiler=self.cta_tiler, coord=(i, j, None), proj=(1, 1, None))
        tCsC = thr_mma.partition_C(sC)
        tCrD = cute.make_fragment_like(tCrC, self.dtype)
        if const_expr(spec["s_in"] is not None):
            tCcC_mn = acc_mn_view(thr_mma.partition_C(cute.make_identity_tensor((self.bM, self.bN))))
            acc_mn, out_mn = acc_mn_view(tCrC), acc_mn_view(tCrD)
            nrow, ncol = cute.size(tCcC_mn.shape[0]), cute.size(tCcC_mn.shape[1])
            cols = [col0 + tCcC_mn[0, c][1] for c in range(ncol)]
            bvals = [mBias[lb, cols[c]] for c in range(ncol)]
            cvals = [mC[lb, cols[c]] for c in range(ncol)]
            for r in cutlass.range_constexpr(nrow):
                row = row0 + tCcC_mn[r, 0][0]
                mean, rstd = stats[spec["s_in"], row, 0], stats[spec["s_in"], row, 1]
                for c in cutlass.range_constexpr(ncol):
                    y = rstd * (acc_mn[r, c] - mean * cvals[c]) + bvals[c]
                    if const_expr(spec["gelu"]):
                        z = y * (0.7978845608028654 + 0.035677408136300125 * y * y)
                        y = 0.5 * (y * cute.math.tanh(z, fastmath=True) + y)
                    out_mn[r, c] = y.to(self.dtype)
        else:
            tCrD.store(tCrC.load().to(self.dtype))
        cute.autovec_copy(tCrD, tCsC)
        cute.arch.sync_threads()
        thr_copy_C = tiled_copy_C.get_slice(tidx)
        tCsC_epi, tCgC_epi = thr_copy_C.partition_S(sC), thr_copy_C.partition_D(gD)
        tCrC_epi = cute.make_fragment_like(tCsC_epi)
        cute.autovec_copy(tCsC_epi, tCrC_epi)
        lanes = self.bN // 8
        rows_per_step = self.num_threads // lanes
        nsteps = cute.size(tCrC_epi, mode=[1])
        if const_expr(spec["aux"]):
            gAux = cute.local_tile(mAux, tiler=self.cta_tiler, coord=(i, j, None), proj=(1, 1, None))
            tCgAux = thr_copy_C.partition_D(gAux)
            tCrAux = cute.make_fragment_like(tCgAux)
            cute.copy(tiled_copy_C, tCgAux, tCrAux)
            colb = col0 + (tidx % lanes) * 8
            bias_r = cute.make_fragment_like(tCrC_epi[None, 0, 0], Float32)
            for e in cutlass.range_constexpr(cute.size(bias_r)):
                bias_r[e] = mBias[lb, colb + e]
            bias8 = bias_r.load()
            for m in cutlass.range_constexpr(nsteps):
                v = tCrC_epi[None, m, 0].load().to(Float32) + bias8 + tCrAux[None, m, 0].load().to(Float32)
                tCrC_epi[None, m, 0] = v.to(self.dtype)
        if const_expr(spec["s_out"] is not None):
            for m in cutlass.range_constexpr(nsteps):
                v = tCrC_epi[None, m, 0].load().to(Float32)
                a = v.reduce(cute.ReductionOp.ADD, Float32.zero, 0)
                b = (v * v).reduce(cute.ReductionOp.ADD, Float32.zero, 0)
                off = lanes // 2
                while off >= 1:
                    a = a + cute.arch.shuffle_sync_bfly(a, offset=off, mask=-1, mask_and_clamp=31)
                    b = b + cute.arch.shuffle_sync_bfly(b, offset=off, mask=-1, mask_and_clamp=31)
                    off = off // 2
                if tidx % lanes == 0:
                    s1[spec["s_out"], row0 + tidx // lanes + rows_per_step * m, j] = a
                    s2[spec["s_out"], row0 + tidx // lanes + rows_per_step * m, j] = b
        cute.copy(tiled_copy_C, tCrC_epi, tCgC_epi)
        cute.arch.sync_threads()
        oi = i // self.nq if const_expr(spec["by_img"]) else i
        self.arrive_publish(spec["out"], l + spec["dl"], oi, spec["expect"], spec["s_out"], row0, Dm, s1, s2, stats, eps, arrive, ready)

    @cute.jit
    def arrive_publish(self, slot: cutlass.Constexpr[int], l: Int32, oi: Int32, expect: cutlass.Constexpr[int],
                       s_out: cutlass.Constexpr, row0: Int32, Dm: cutlass.Constexpr[int], s1, s2, stats, eps: Float32, arrive, ready):
        oidx = self.fidx(slot, l, oi)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            lane = cute.arch.lane_idx()
            old = Int32(0)
            if lane == 0:
                old = cute.arch.atomic_add(arrive.iterator + oidx, Int32(1), sem="acq_rel", scope="gpu")
            old = cute.arch.shuffle_sync(old, 0)
            if old == expect - 1:
                if const_expr(s_out is not None):
                    cute.arch.load(arrive.iterator + oidx, Int32, sem="acquire", scope="gpu")
                    nt = Dm // self.bN
                    inv_d = Float32(1.0 / Dm)
                    for rr in cutlass.range_constexpr(self.bM // 32):
                        frow = row0 + lane + 32 * rr
                        a, b = Float32(0.0), Float32(0.0)
                        for tt in cutlass.range_constexpr(nt):
                            a = a + s1[s_out, frow, tt]
                            b = b + s2[s_out, frow, tt]
                        mu = a * inv_d
                        vr = cutlass.max(b * inv_d - mu * mu, Float32(0.0))
                        stats[s_out, frow, 0] = mu
                        stats[s_out, frow, 1] = cute.math.rsqrt(vr + eps, fastmath=True)
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.store(ready.iterator + oidx, Int32(1), sem="release", scope="gpu")

    @cute.jit
    def run_attn(self, l: Int32, i: Int32, j: Int32, mQKV, mO, key_len, arrive, ready, sQ, sK, sV, sQ_layout,
                 tiled_copy_QKV, tiled_copy_O, tiled_mma, tiled_mma_pv, tidx: Int32):
        H, hd, nq, kvb = self.H, self.hd, self.nq, self.kv_block
        b, h = i // H, i % H
        m = b * nq + j
        self.wait_ready(ready, self.fidx(SLOT_QKV, l, b), tidx)
        gQ = cute.local_tile(mQKV, (self.bM, hd), (m, h))
        gK = cute.local_tile(mQKV, (kvb, hd), (None, H + h))
        gV = cute.local_tile(mQKV, (kvb, hd), (None, 2 * H + h))
        gO = cute.local_tile(mO, (self.bM, hd), (m, h))
        kv0 = b * (self.S // kvb)
        klen = key_len[b]
        n_blocks = (klen + kvb - 1) // kvb
        sVt = cute.composition(sV, cute.make_layout((hd, kvb), stride=(kvb, 1)))
        cp = tidx < 2 * kvb
        thr_copy = tiled_copy_QKV.get_slice(tidx % (2 * kvb))
        tQgQ, tQsQ = thr_copy.partition_S(gQ), thr_copy.partition_D(sQ)
        tKgK, tKsK = thr_copy.partition_S(gK), thr_copy.partition_D(sK)
        tVgV, tVsV = thr_copy.partition_S(gV), thr_copy.partition_D(sV)
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        thr_pv = tiled_mma_pv.get_slice(tidx)
        tOrVt = thr_pv.make_fragment_B(thr_pv.partition_B(sVt))
        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.bM, hd)), Float32)
        acc_Oh = cute.make_rmem_tensor(thr_pv.partition_shape_C((self.bM, hd)), Float16)
        acc_O.fill(0.0)
        s2r_Q = cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), self.dtype), tiled_mma)
        s2r_K = cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), self.dtype), tiled_mma)
        s2r_V = cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), self.dtype), tiled_mma_pv)
        thr_Q, thr_K, thr_V = s2r_Q.get_slice(tidx), s2r_K.get_slice(tidx), s2r_V.get_slice(tidx)
        P = SimpleNamespace(
            tiled_mma=tiled_mma, thr_mma=thr_mma, tiled_mma_pv=tiled_mma_pv, tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt,
            acc_O=acc_O, acc_Oh=acc_Oh,
            tiled_copy_QKV=tiled_copy_QKV, tKgK=tKgK, tKsK=tKsK, tVgV=tVgV, tVsV=tVsV, kv0=kv0, klen=klen, cp=cp,
            s2r_Q=s2r_Q, s2r_K=s2r_K, s2r_V=s2r_V,
            tSsQ=thr_Q.partition_S(sQ), tSrQ_cv=thr_Q.retile(tSrQ),
            tSsK=thr_K.partition_S(sK), tSrK_cv=thr_K.retile(tSrK),
            tOsVt=thr_V.partition_S(sVt), tOrVt_cv=thr_V.retile(tOrVt),
            row_max=cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), Float32),
            row_sum=cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), Float32),
            scale_log2=Float32(self.softmax_scale_log2))
        P.row_max.fill(-Float32.inf)
        P.row_sum.fill(0.0)
        n_block = n_blocks - 1
        if cp:
            cute.copy(tiled_copy_QKV, tQgQ, tQsQ)
            cute.copy(tiled_copy_QKV, tKgK[None, None, None, kv0 + n_block], tKsK)
        cute.arch.cp_async_commit_group()
        self.attn_block(P, n_block, is_first=True, masked=True)
        for nb in range(1, n_blocks):
            self.attn_block(P, n_blocks - 1 - nb, is_first=False, masked=False)
        acc_O_mn = acc_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(P.row_sum)):
            rs = quad_reduce(P.row_sum[r], lambda x, y: x + y)
            inv = 1.0 if (rs == 0.0 or rs != rs) else cute.arch.rcp_approx(rs)
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        sO = cute.make_tensor(sQ.iterator, sQ_layout)
        smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype)
        thr_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        cute.copy(smem_copy_atom_O, thr_O.retile(rO), thr_O.partition_D(sO))
        cute.arch.sync_threads()
        thr_copy_O = tiled_copy_O.get_slice(tidx % (2 * kvb))
        tOsO, tOgO = thr_copy_O.partition_S(sO), thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self.dtype)
        if cp:
            cute.copy(tiled_copy_O, tOsO, tOrO)
            cute.copy(tiled_copy_O, tOrO, tOgO)
        cute.arch.sync_threads()
        self.arrive_publish(SLOT_ATTN, l, m, H, None, Int32(0), self.D, None, None, None, Float32(0.0), arrive, ready)

    @cute.jit
    def attn_block(self, P, n_block: Int32, is_first: cutlass.Constexpr[bool], masked: cutlass.Constexpr[bool]):
        kvb = self.kv_block
        acc_S = cute.make_rmem_tensor(P.thr_mma.partition_shape_C((self.bM, kvb)), Float32)
        acc_S.fill(0.0)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if P.cp:
            cute.copy(P.tiled_copy_QKV, P.tVgV[None, None, None, P.kv0 + n_block], P.tVsV)
        cute.arch.cp_async_commit_group()
        nk = cute.size(P.tSsQ.shape[2])
        cute.copy(P.s2r_Q, P.tSsQ[None, None, 0], P.tSrQ_cv[None, None, 0])
        cute.copy(P.s2r_K, P.tSsK[None, None, 0], P.tSrK_cv[None, None, 0])
        for k in cutlass.range_constexpr(nk):
            k_next = (k + 1) % nk
            cute.copy(P.s2r_Q, P.tSsQ[None, None, k_next], P.tSrQ_cv[None, None, k_next])
            cute.copy(P.s2r_K, P.tSsK[None, None, k_next], P.tSrK_cv[None, None, k_next])
            cute.gemm(P.tiled_mma, acc_S, P.tSrQ[None, None, k], P.tSrK[None, None, k], acc_S)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if n_block > 0:
            if P.cp:
                cute.copy(P.tiled_copy_QKV, P.tKgK[None, None, None, P.kv0 + n_block - 1], P.tKsK)
            cute.arch.cp_async_commit_group()
        acc_S_mn = acc_mn_view(acc_S)
        acc_O_mn = acc_mn_view(P.acc_O)
        if const_expr(masked):
            tScS_mn = acc_mn_view(P.thr_mma.partition_C(cute.make_identity_tensor((self.bM, kvb))))
            col_base = n_block * kvb
        for r in cutlass.range_constexpr(cute.size(P.row_max)):
            if const_expr(masked):
                for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                    if col_base + tScS_mn[0, c][1] >= P.klen:
                        acc_S_mn[r, c] = -Float32.inf
            row = acc_S_mn[r, None].load()
            mx = quad_reduce(row.reduce(cute.ReductionOp.MAX, -Float32.inf, 0), lambda x, y: cute.arch.fmax(x, y))
            if const_expr(not is_first):
                mx = cute.arch.fmax(P.row_max[r], mx)
            row_exp = cute.math.exp2(row * P.scale_log2 - mx * P.scale_log2, fastmath=True)
            rs = row_exp.reduce(cute.ReductionOp.ADD, Float32.zero, 0)
            if const_expr(not is_first):
                corr = cute.math.exp2(P.row_max[r] * P.scale_log2 - mx * P.scale_log2, fastmath=True)
                rs = rs + P.row_sum[r] * corr
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * corr
            P.row_max[r] = mx
            P.row_sum[r] = rs
            acc_S_mn[r, None] = row_exp
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        ld = cute.logical_divide(rP.layout, (None, None, 2))
        tOrS = cute.make_tensor(rP.iterator, cute.make_layout(
            ((ld.shape[0], ld.shape[2][0]), ld.shape[1], ld.shape[2][1]),
            stride=((ld.stride[0], ld.stride[2][0]), ld.stride[1], ld.stride[2][1])))
        nkv = cute.size(tOrS.shape[2])
        P.acc_Oh.fill(0.0)
        cute.copy(P.s2r_V, P.tOsVt[None, None, 0], P.tOrVt_cv[None, None, 0])
        for k in cutlass.range_constexpr(nkv):
            k_next = (k + 1) % nkv
            cute.copy(P.s2r_V, P.tOsVt[None, None, k_next], P.tOrVt_cv[None, None, k_next])
            cute.gemm(P.tiled_mma_pv, P.acc_Oh, tOrS[None, None, k], P.tOrVt[None, None, k], P.acc_Oh)
        P.acc_O.store(P.acc_O.load() + P.acc_Oh.load().to(Float32))

    def _make_smem_layout_AB(self, dtype, copy_bits, smem_tiler):
        major = min(smem_tiler[1], 128 * 8 // dtype.width)
        swizzle_bits = min(int(math.log2(major * dtype.width // copy_bits)), 3)
        swizzle = cute.make_swizzle(swizzle_bits, int(math.log2(copy_bits // 8)), int(math.log2(copy_bits // dtype.width)))
        atom = cute.make_layout((8, major), stride=(major, 1))
        return cute.tile_to_shape(atom, smem_tiler, (0, 1, 2)), swizzle

    def _make_smem_layout_C(self, dtype, copy_bits, smem_tiler):
        major = smem_tiler[1]
        swizzle_bits = min(int(math.log2(major * dtype.width // copy_bits)), 3)
        atom = cute.make_composed_layout(cute.make_swizzle(swizzle_bits, 3, int(math.log2(major)) - 3), 0,
                                         cute.make_layout((8, major), stride=(major, 1)))
        return cute.tile_to_shape(atom, smem_tiler, (0, 1))

    def _make_gmem_tiled_copy(self, atom_copy, dtype, inner, copy_bits):
        copy_elems = copy_bits // dtype.width
        dim1 = inner // copy_elems
        thread_layout = cute.make_layout((self.num_threads // dim1, dim1), stride=(dim1, 1))
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, cute.make_layout((1, copy_elems)))


def fold_ln(W, b, g, beta, dtype):
    Wp = (W.float() * g.float()[None]).to(dtype).contiguous()
    return Wp, Wp.float().sum(1).contiguous(), (b.float() + W.float() @ beta.float()).contiguous()


def gemm_tasks(op, l, nm, nn, group_m):
    return [(op, l, i, j) for g in range(0, nm, group_m) for j in range(nn) for i in range(g, min(g + group_m, nm))]


class Engine:
    @torch.no_grad()
    def __init__(self, vm, B, tile_n, stages, warps, fold_k, group_m, kv_block=64):
        cfg, emb = vm.config, vm.embeddings
        D, F0, H, L, p = cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_hidden_layers, cfg.patch_size
        up = lambda n, m: (n + m - 1) // m * m
        S0, Kp0, hd0 = emb.num_patches, 3 * p * p, D // H
        S, Kp, Fd, hd = up(S0, 128), up(Kp0, 32), up(F0, 128), up(hd0, 16)
        tm, tn = 128, tile_n
        assert D % tn == 0 and D % 32 == 0
        self.vm, self.B, self.S, self.S0, self.D, self.Kp, self.Kp0, self.p, self.eps = vm, B, S, S0, D, Kp, Kp0, p, cfg.layer_norm_eps
        self.conv = isinstance(emb.patch_embedding, torch.nn.Conv2d)
        M, nm, nq = B * S, B * S // tm, S // tm
        self.M = M
        dev = next(vm.parameters()).device
        self.dev = dev
        dt = torch.float16
        hf = dict(device=dev, dtype=dt)
        st = lambda xs: torch.stack(xs).contiguous()
        ph = lambda w: F.pad(w.reshape(H, hd0, -1), (0, 0, 0, hd - hd0)).reshape(H * hd, -1)
        Wqkv, cqkv, bqkv, Wo, bo, W1, c1, b1, W2, b2 = ([] for _ in range(10))
        for lyr in vm.encoder.layers:
            a, mlp = lyr.self_attn, lyr.mlp
            Wc = torch.cat([ph(a.q_proj.weight), ph(a.k_proj.weight), ph(a.v_proj.weight)])
            bc = torch.cat([ph(a.q_proj.bias).flatten(), ph(a.k_proj.bias).flatten(), ph(a.v_proj.bias).flatten()])
            w, c, bp = fold_ln(Wc, bc, lyr.layer_norm1.weight, lyr.layer_norm1.bias, dt)
            Wqkv += [w]; cqkv += [c]; bqkv += [bp]
            Wo += [F.pad(a.out_proj.weight.view(D, H, hd0), (0, hd - hd0)).reshape(D, H * hd).to(dt)]; bo += [a.out_proj.bias.float()]
            w, c, bp = fold_ln(F.pad(mlp.fc1.weight, (0, 0, 0, Fd - F0)), F.pad(mlp.fc1.bias, (0, Fd - F0)), lyr.layer_norm2.weight, lyr.layer_norm2.bias, dt)
            W1 += [w]; c1 += [c]; b1 += [bp]
            W2 += [F.pad(mlp.fc2.weight, (0, Fd - F0)).to(dt)]; b2 += [mlp.fc2.bias.float()]
        pe = emb.patch_embedding
        Wpe = F.pad(pe.weight.reshape(D, Kp0), (0, Kp - Kp0)).reshape(1, D, Kp).to(dt).contiguous()
        bpe = pe.bias.float().reshape(1, D).contiguous()
        self.patches = torch.empty(M, Kp, **hf)
        self.x_a, self.x_b = (torch.empty(M, D, **hf) for _ in range(2))
        self.attn, self.qkv, self.h = torch.empty(M, H * hd, **hf), torch.empty(M, 3 * H * hd, **hf), torch.empty(M, Fd, **hf)
        self.pos = torch.zeros(M, D, **hf)
        self.embeds = torch.empty(B, D, **hf)
        self.key_pad = torch.zeros(B, S, dtype=torch.bool, device=dev)
        self.key_pad[:, S0:] = True
        self.key_len = torch.full((B,), S0, dtype=torch.int32, device=dev)
        if self.conv:
            self.pos.view(B, S, D)[:, :S0] = emb.position_embedding.weight.to(dt)
        else:
            side = int(S0 ** 0.5)
            self.pos_grid = emb.position_embedding.weight.detach().to(dt).reshape(side, side, D).permute(2, 0, 1)[None].contiguous()
        self.stats = torch.zeros(2, M, 2, device=dev)
        self.s1 = torch.zeros(2, M, D // tn, device=dev)
        self.s2 = torch.zeros(2, M, D // tn, device=dev)
        nflag = NSLOT * (L + 1) * nm
        self.arrive = torch.zeros(nflag, dtype=torch.int32, device=dev)
        self.ready = torch.zeros(nflag + 1, dtype=torch.int32, device=dev)
        self.task_ctr = torch.zeros(1, dtype=torch.int32, device=dev)
        T = []
        add = lambda op, l, nn: T.extend(gemm_tasks(op, l, nm, nn, group_m))
        add(OP_EMB, 0, D // tn)
        for l in range(L):
            add(OP_QKV, l, 3 * H * hd // tn)
            T.extend((OP_ATTN, l, b * H + h, q) for b in range(B) for h in range(H) for q in range(nq))
            add(OP_OPROJ, l, D // tn); add(OP_FC1, l, Fd // tn); add(OP_FC2, l, D // tn)
        dep_slot = {OP_EMB: None, OP_QKV: SLOT_X, OP_OPROJ: SLOT_ATTN, OP_FC1: SLOT_XMID, OP_FC2: SLOT_H, OP_ATTN: SLOT_QKV}
        rows = []
        for (op, l, i, j) in T:
            slot = dep_slot[op]
            ii = i // H if op == OP_ATTN else i
            rows.append((op, l, i, j, nflag if slot is None else (slot * (L + 1) + l) * nm + ii))
        self.tasks = torch.tensor(rows, dtype=torch.int32, device=dev)
        dummy_c = torch.zeros(1, 1, device=dev)
        A = [self.patches, self.x_a, self.attn, self.x_b, self.h]
        Wt = [Wpe, st(Wqkv), st(Wo), st(W1), st(W2)]
        Dt = [self.x_a, self.qkv, self.x_b, self.h, self.x_a]
        AUX = [self.pos, self.x_a, self.x_a, self.x_a, self.x_b]
        BIAS = [bpe, st(bqkv), st(bo), st(b1), st(b2)]
        CV = [dummy_c, st(cqkv), dummy_c, st(c1), dummy_c]
        self.torch_args = A + Wt + Dt + AUX + BIAS + CV + [self.stats, self.s1, self.s2, self.key_len, self.tasks,
                                                           self.task_ctr, self.arrive, self.ready]
        self.cute_args = [from_dlpack(t, assumed_align=16) for t in self.torch_args]
        ctas_per_sm = 2 if warps == 4 else 1
        sep_c = ctas_per_sm == 1 and stages * (tm + tn) * 32 * 2 + tm * tn * 2 <= 99 * 1024
        self.kern = MegaSm86(D, Fd, S, H, hd, hd0, L, nm, tn, stages, warps, fold_k, kv_block, sep_c)
        self.grid = torch.cuda.get_device_properties(dev).multi_processor_count * ctas_per_sm
        self.compiled = cute.compile(self.kern, *self.cute_args, Int32(0), Int32(0), Float32(self.eps), Int32(self.grid),
                                     cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        self.graph = None

    def _launch(self):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled(*self.cute_args, Int32(0), Int32(self.tasks.shape[0]), Float32(self.eps), Int32(self.grid), stream)

    @torch.no_grad()
    def _graph_body(self):
        self.arrive.zero_(); self.ready[:-1].zero_(); self.ready[-1] = 1; self.task_ctr.zero_()
        self._launch()
        vm, hd = self.vm, self.vm.head
        x = F.layer_norm(self.x_a, (self.D,), vm.post_layernorm.weight, vm.post_layernorm.bias, self.eps).view(self.B, self.S, self.D)
        y = hd.attention(hd.probe.expand(self.B, -1, -1), x, x, key_padding_mask=self.key_pad, need_weights=False)[0]
        self.embeds.copy_((y + hd.mlp(hd.layernorm(y)))[:, 0])

    @torch.no_grad()
    def _prepare(self, inp):
        x = inp["pixel_values"].to(self.dev, torch.float16)
        if self.conv:
            p = self.p
            x = x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).reshape(self.B, self.S0, self.Kp0)
        self.patches.copy_(F.pad(x, (0, self.Kp - self.Kp0, 0, self.S - self.S0)).reshape(self.M, self.Kp))
        if "spatial_shapes" in inp:
            mask = inp["pixel_attention_mask"]
            self.key_pad.copy_(F.pad(mask == 0, (0, self.S - self.S0), value=True))
            self.key_len.copy_(mask.sum(1))
            pos = self.pos.view(self.B, self.S, self.D)
            shapes = [tuple(s) for s in inp["spatial_shapes"].tolist()]
            for hw in set(shapes):
                emb = F.interpolate(self.pos_grid, size=hw, mode="bilinear", align_corners=False, antialias=True)
                emb = emb[0].flatten(1).t()
                idx = [k for k, s in enumerate(shapes) if s == hw]
                pos[idx, : len(emb)] = emb
                pos[idx, len(emb):] = emb[0]

    @torch.no_grad()
    def forward(self, inp):
        self._prepare(inp)
        if self.graph is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._graph_body()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._graph_body()
        self.graph.replay()
        return self.embeds


class Siglip2Embedder:
    CONFIGS = [dict(warps=w, tile_n=t, stages=s, fold_k=f, group_m=g)
               for w, t in ((8, 128), (4, 64)) for s in (3, 4) for f in (2, 4) for g in (8, 16)]
    CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mega_sm86_tune.json")

    def __init__(self, model_id="google/siglip2-base-patch16-naflex", batch_size=64, device="cuda"):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.vm = AutoModel.from_pretrained(model_id, dtype=torch.float16, device_map=device).vision_model.eval()
        self.B, self.device = batch_size, device
        self.engine = Engine(self.vm, batch_size, **self._tune(model_id))

    def _time(self, engine, inp, iters=10):
        for _ in range(3):
            engine.forward(inp)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            engine.forward(inp)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    def _tune(self, model_id):
        key = f"{torch.cuda.get_device_name()}|{model_id}|{self.B}"
        cache = json.load(open(self.CACHE)) if os.path.exists(self.CACHE) else {}
        if key not in cache:
            cfg, B = self.vm.config, self.B
            if isinstance(self.vm.embeddings.patch_embedding, torch.nn.Conv2d):
                inp = dict(pixel_values=torch.randn(B, 3, cfg.image_size, cfg.image_size))
            else:
                S = cfg.num_patches
                inp = dict(pixel_values=torch.randn(B, S, 3 * cfg.patch_size ** 2), pixel_attention_mask=torch.ones(B, S, dtype=torch.long),
                           spatial_shapes=torch.tensor([[int(S ** 0.5)] * 2] * B))
            with torch.no_grad():
                ref = self.vm(**{k: v.to(self.device) for k, v in inp.items()}).pooler_output.float()
            timed = []
            for c in self.CONFIGS:
                engine = Engine(self.vm, B, **c)
                out = engine.forward(inp).float()
                if F.cosine_similarity(out, ref, dim=-1).min() > 0.999:
                    timed.append((self._time(engine, inp), c))
                del engine
                torch.cuda.empty_cache()
            cache[key] = min(timed, key=lambda t: t[0])[1]
            json.dump(cache, open(self.CACHE, "w"), indent=1)
        return cache[key]

    @torch.no_grad()
    def embed(self, images):
        inp = dict(self.processor(images=images, return_tensors="pt"))
        n_all = len(inp["pixel_values"])
        out = []
        for k in range(0, n_all, self.B):
            n = min(self.B, n_all - k)
            pad = lambda t: torch.cat([t[k:k + n], t[k + n - 1:k + n].expand(self.B - n, *t.shape[1:])])
            out.append(self.engine.forward({key: pad(t) for key, t in inp.items()})[:n].clone())
        return torch.cat(out)