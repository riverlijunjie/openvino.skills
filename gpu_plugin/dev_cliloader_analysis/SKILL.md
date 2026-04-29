---
name: dev_cliloader_analysis
description: Analyze cliloader log for performance regression issues of GPU plugin inference.
---

When analyze inference performance regression issues of GPU plugin using cliloader log, the following steps can be taken to identify the root cause and optimize the code:

1. **Top priority policies**
   - Don't create new commit, only modify code locally and remote copy to remote machine for build and test.
   - Don't push code to remote repository.
   - Follow the same code style and conventions used in the existing codebase to maintain consistency.
   - Exit task if remote machine is not accessible or if there are any issues with remote build and test.

2. **Read log and analyze kernel performance**
    - call classify_kernels.py to classify kernels into different categories such as GEMM, Attention, MOE, etc.
    - call analyze_perf.py to analyze the performance of each kernel category and identify the bottlenecks.

3. **Remote build and test**
    - After identifying the bottlenecks, apply optimizations to the code locally.
    - Remote machine: gta@10.36.24.51, pwd:gta
    - Remote copy the modified code to the remote machine for build and test.
    - Remote build OpenVINO with the optimized code:
        - cd ~/rivera/openvino
        - build command: source ./build.sh
    - Remote test:
        - cd ~/river/openvino.genai/tools/llm_bench
        - test command: source ./runme.sh 
    - Get output log to analyze first token latency and token rate

4. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance to "SUMMARY.md"
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.


