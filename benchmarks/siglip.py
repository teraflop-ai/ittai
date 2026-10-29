import sys
import requests
import torch
from PIL import Image
from ittai.siglip.kernel import Siglip2Embedder

model_id = sys.argv[1] if len(sys.argv) > 1 else "google/siglip2-so400m-patch14-384"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 64
img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
emb = Siglip2Embedder(model_id, batch_size=B)
inp = dict(emb.processor(images=[img] * B, return_tensors="pt"))
kw = {k: v.cuda() for k, v in inp.items()}
vm_c = torch.compile(emb.vm, mode="max-autotune", dynamic=False)


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
    t_eager = ms(lambda: emb.vm(**kw))
    t_comp = ms(lambda: vm_c(**kw))
    ref = vm_c(**kw).pooler_output.float()
t_mega = ms(lambda: emb.engine.forward(inp))
cos = torch.nn.functional.cosine_similarity(emb.engine.forward(inp).float(), ref, dim=-1).min().item()
c, S = emb.vm.config, emb.engine.S0
flops = B * c.num_hidden_layers * (2 * S * c.hidden_size * (4 * c.hidden_size + 2 * c.intermediate_size) + 4 * S * S * c.hidden_size)
for n, t in (("hf eager", t_eager), ("hf max-autotune", t_comp), ("mega", t_mega)):
    print(f"{n:16s} {t:8.2f} ms  {flops / t / 1e9:6.1f} TFLOPS  {t_comp / t:.2f}x vs compile")
print(f"{model_id}  B={B}  {B / t_mega * 1e3:.1f} img/s  min cos vs compile {cos:.5f}")