import json
import os

import torch
import torch.nn.functional as F

from .config import *
from .engine import Engine
from .sm80 import MegaSm80
from .sm90 import MegaSm90


class Qwen3Embedder:
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
            for c in CONFIGS[self.arch]:
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