import os
base = (
    "In modern deep learning inference, the attention mechanism dominates the cost of "
    "decoding long sequences because every newly generated token must attend to the entire "
    "key-value cache accumulated so far. As the context length grows, the memory traffic "
    "required to stream the cached keys and values from device memory becomes the primary "
    "bottleneck, especially on bandwidth-limited integrated and discrete GPUs. Engineers "
    "therefore spend considerable effort tuning the work-group tiling, the subgroup layout, "
    "and the unroll factors of the scaled-dot-product-attention kernel so that the hardware "
    "executes at the highest possible fraction of its theoretical memory bandwidth. "
)
# Repeat to reach a long context; vary the index so it is not pure repetition.
parts = []
for i in range(60):
    parts.append(f"Section {i+1}. " + base)
prompt = "".join(parts) + "\nGiven all of the above, summarize the key bottleneck and the main optimization levers in detail."

with open(r"D:\river\workspace\dev_sdpa_micro_opt\long_prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

print("chars", len(prompt))
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(r"D:\river\models\qwen3-8b\pytorch\ov\OV_FP16-4BIT_DEFAULT")
    n = len(tok(prompt)["input_ids"])
    print("tokens", n)
except Exception as e:
    print("tokenizer_skip", repr(e))
