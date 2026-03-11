---
name: dev_qwen3.5
description: Develop Qwen3.5 MoE for better performance. Use when working on Qwen3.5 MoE models or improving MoE operation efficiency.
---

When develop qwen3.5 moe feature, always include:
1. **Read pytorch code**: Read Qwen3.5 MoE code. This includes:
    - https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
2. **Read C++ code**: Read Qwen3.5 MoE equivalent code. This includes:
    - openvino.genai/src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_moe.cpp
    - openvino.genai/src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_moe.hpp
    - openvino.genai/src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_text.cpp
    - openvino.genai/src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_text.hpp
3. **Identify optimization opportunities**: Look for areas in the code where performance can be improved, such as inefficient algorithms, redundant computations, or memory bottlenecks.
4. **Implement optimizations**: Apply code changes to optimize the identified areas. This may include algorithmic improvements, better memory management, or leveraging hardware capabilities.
5. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance.
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.