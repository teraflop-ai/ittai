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

CONFIGS = {
    "sm90": [dict(tile_n=t, stages=s, kv_block=k, group_m=g) for t, s in ((256, 3), (128, 4), (128, 5)) for k in (64, 128) for g in (8, 16)],
    "sm80": [dict(warps=w, tile_n=t, stages=s, kv_block=k, group_m=g)
             for w, t, s in ((4, 128, 4), (8, 128, 4), (8, 128, 5), (8, 256, 3), (8, 256, 4))
             for k in (64, 128) for g in (8, 16) if w == 8 or k == 64],
}