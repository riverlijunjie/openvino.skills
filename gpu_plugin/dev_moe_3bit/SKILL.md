---
name: moe_3bit_optimization
description: >
  Unified skill for 3bit support to Mixture-of-Experts (MoE) GPU optimization in OpenVINO.
---

## Workflow

1. **Read references**: read material related to 3bit compression techniques and summarize the key points:
    tickets: https://jira.devtools.intel.com/browse/CVS-180191
              https://jira.devtools.intel.com/browse/CVS-182951
    papers:
        BitNet: https://arxiv.org/pdf/2504.12285v1  (training-based method)
        QUIP: https://arxiv.org/abs/2402.04396 
        AQLM: https://arxiv.org/pdf/2401.06118
        SignRoundv2: https://arxiv.org/abs/2512.04746
        QTIP: https://arxiv.org/abs/2406.11235 
        YAQA: https://github.com/Cornell-RelaxML/yaqa-quantization
        LoftQ: https://arxiv.org/pdf/2310.08659
        PiSSA: https://arxiv.org/pdf/2404.02948
        IR-QLoRA: https://arxiv.org/pdf/2402.05445
        ParetoQ: https://arxiv.org/pdf/2502.02631
        LR-QAT: https://arxiv.org/pdf/2406.06385
                https://github.com/qualcomm-ai-research/LR-QAT
        ApiQ: https://arxiv.org/pdf/2402.05147
        
    gguf:
        https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main

    more material:
       LoftQ establishes a principled SVD-based initialization for LoRA adapters under low-bit quantization. Rather than the standard LoRA init — Gaussian noise for A and zeros for B, which makes AB = 0 at the start and causes gradients for A to be zero and for B to be random — LoftQ initializes A and B from the true quantization residual W − Q(W). This dramatically improves fine-tuning convergence, especially at ≤ 4-bit. Applicable for W_q = FQ(W) + BA, in NNCF W_q = FQ(W + BA).

        PiSSA takes a complementary approach: instead of initializing adapters from the residual, it initializes them from the principal singular components of the original weight, leaving only the residual W − BA to be quantized. Limitations: the adapter BA remains in full precision at inference, the final model is quant(W_res) + BA.

        IR-QLoRA addresses both initialization and quantization grid quality (NFx quantization). For quantization, it introduces Information Calibration Quantization (ICQ), which adds a learnable calibration constant τ into the NormalFloat quantization function. For adapter initialization, IR-QLoRA similarly uses the LoftQ-style residual SVD. Gradient Clipping: Maximum gradient norm is limited to 0.3, Rank=64, dropout=0.1, dataset https://huggingface.co/datasets/tatsu-lab/alpaca 10000 steps, bs=16, lr=2e-4.

        Quantization 

        ParetoQ For 3-bit and 4-bit specifically, ParetoQ leverages Learned Step Size Quantization (LSQ), which is more effective when the zero value is included in the set of representable output levels (symmetric quantization grids).

        LR-QAT, GitHub trains low-rank adapters jointly with learned quantization step sizes in a fake-quantize framework. The forward pass applies quantization to W₀ + (α/r)AB directly:

5. **Document**: Update `SUMMARY.md` with results; keep SKILL.md as concise reference

### Rules
- Follow existing code style and conventions
- Don't create new commits or push code until optimization is verified
- Don't modify oneDNN code directly
- Include before-and-after performance metrics and roofline efficiency ratios in documentation

---

## Related Docs

