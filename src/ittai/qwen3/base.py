import math

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, BFloat16, const_expr

from .config import *


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