import json
import random

# Generate a ShareGPT-format dataset for continuous_batching_benchmark.
# Each conversation: one human turn (long-ish prompt) + one gpt turn (reference).
# Long inputs + many prompts -> high concurrency -> attention takes a larger
# share of decode time, which is where the wg_n=1 PA-decode optimization helps.

random.seed(1234)

TOPICS = [
    "transformer attention", "GPU memory bandwidth", "KV cache management",
    "tensor parallelism", "quantization of weights", "speculative decoding",
    "operator fusion", "roofline analysis", "paged attention", "MoE routing",
    "kernel autotuning", "warp scheduling", "register pressure", "L3 cache reuse",
]

SENT = [
    "The decode phase of an LLM is dominated by memory traffic rather than raw compute.",
    "Each generated token must read the entire key/value cache for every attention head.",
    "As the context grows, the attention kernel reads progressively more data per step.",
    "Work-group tiling decides how many key columns each sub-group processes at once.",
    "Choosing a tile that matches the number of new query tokens avoids wasted lanes.",
    "On Xe3 the paged-attention decode path usually has only a handful of new queries.",
    "A smaller column tile removes masked work and improves effective bandwidth.",
    "Prefill, in contrast, processes the whole prompt and prefers a wider tile.",
    "Continuous batching interleaves many sequences to keep the GPU busy.",
    "Higher concurrency raises the relative cost of the attention operator.",
]


def make_prompt(min_sent=40, max_sent=70):
    topic = random.choice(TOPICS)
    n = random.randint(min_sent, max_sent)
    body = " ".join(random.choice(SENT) for _ in range(n))
    return (f"Please analyze the following notes about {topic} and write a concise "
            f"technical summary highlighting the main performance bottleneck.\n\n{body}")


N = 200
data = []
for i in range(N):
    human = make_prompt()
    gpt = ("The dominant bottleneck is memory bandwidth in the attention kernel, "
           "which grows with context length and concurrency. " * 8)
    data.append({
        "id": f"synthetic_{i}",
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": gpt},
        ],
    })

out = r"D:\river\workspace\dev_sdpa_micro_opt\cb_dataset.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f)
print("wrote", out, "conversations", len(data))
