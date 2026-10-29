import numpy as np
import torch
import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from .config import *
from .sm80 import MegaSm80
from .sm90 import MegaSm90


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