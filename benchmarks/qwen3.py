import sys
import torch
import torch.nn.functional as F
from ittai.qwen3.kernel import Qwen3Embedder, D, DQ, DQKV, DGU, FF, L

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-Embedding-0.6B"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 64
S = int(sys.argv[3]) if len(sys.argv) > 3 else 512
emb = Qwen3Embedder(model_id, max_tokens=B * S, max_seqs=B)
text = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: " + "What is the capital of China? " * S
ids = [emb.tok(text, truncation=True, max_length=S)["input_ids"]] * B
kw = dict(input_ids=torch.tensor(ids).cuda(), attention_mask=torch.ones(B, len(ids[0]), dtype=torch.long).cuda())
model_c = torch.compile(emb.model, mode="max-autotune", dynamic=False)


def ms(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[iters // 2]


with torch.inference_mode():
    t_eager = ms(lambda: emb.model(**kw))
    t_comp = ms(lambda: model_c(**kw))
    ref = F.normalize(model_c(**kw).last_hidden_state[:, -1].float(), dim=-1)
t_mega = ms(lambda: emb.engine.forward(ids))
cos = F.cosine_similarity(emb.engine.forward(ids), ref, dim=-1).min().item()
T = sum(map(len, ids))
flops = L * (2 * T * D * (DQKV + DQ + DGU + FF) + 2 * sum(len(s) ** 2 for s in ids) * DQ)
for n, t in (("hf eager", t_eager), ("hf max-autotune", t_comp), ("mega", t_mega)):
    print(f"{n:16s} {t:8.2f} ms  {flops / t / 1e9:6.1f} TFLOPS  {t_comp / t:.2f}x vs compile")
print(f"{model_id}  B={B} S={len(ids[0])}  {T / t_mega:.0f}k tok/s  min cos vs compile {cos:.5f}")