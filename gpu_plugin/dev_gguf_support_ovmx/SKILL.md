---
name: dev_gguf_support
description: Develop and optimize GGUF support for better performance and flexibility. 
---

When developing GGUF support feature, always include:
1. **Read documentation**: Read GGUF support related documentation to understand the feature requirements and design. This includes:
    - SUMMARY.md
    - SPEC.md

2. **Local repos**:
    - OpenVINO repo: /home/ov2022/workspace/ovmx/openvino.mx
    - OpenVINO GenAI repo: /home/ov2022/workspace/ovmx/openvino.genai.mx

3. **Implementation**:
    - Implement the GGUF support feature in the local openvino.mx and openvino.genai.mx repos.
    - Ensure that the implementation follows the design and requirements specified in the documentation.


4. **Remote build and validation**:
    - Remote machine: openvino-ci-74@10.239.140.155, pwd: openvino
    - Remote openvino directory: /mnt/river/ovmx/openvino.pipeline.mx/thirdparty/openvino
    - Remote openvino genai directory: /mnt/river/ovmx/openvino.pipeline.mx/thirdparty/openvino.genai
    - Remote copy modified files to remote machine's openvino and openvino.genai directory.
    - Remote build command:
          openvino: cd /mnt/river/ovmx/openvino.pipeline.mx/thirdparty/openvino && source build_release.sh
          openvino.genai: cd /mnt/river/ovmx/openvino.pipeline.mx/thirdparty/openvino.genai && source build.sh
    - Remote unit test command: cd /mnt/river/ovmx/openvino.pipeline.mx/thirdparty/openvino/bin/intel64/Release && ./ov_gpu_unit_tests --gtest_filter=*gguf*
    - Remote test environment setup: cd /mnt/river/ovmx && source venv/bin/activate
    - Remote e2e test command: cd /mnt/river/ovmx/openvino.pipeline.mx/thirdparty/openvino.genai/tools/llm_bench && source run_gguf_test.sh
    - Test models: /mnt/river/moe/models/Qwen3-8B-Q5_K_M.gguf or https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q5_K_M.gguf
    - cliloader is loacted in: C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe

5. **E2E testing**:
    - Run E2E tests to validate the GGUF support feature and ensure it works as expected with the test models.


6. **Profile performance and optimize OpenCL kernel**:
    - Use profiling tools to identify performance bottlenecks in the OpenCL kernel.
        - test app: genai benchmarks
        - profiling tools: cliloader, vtune, etc.
        - collect kernels' latency for prefilling and decoding stages
        - figure out each kerne's roofline based on hardare capabilities, and point out if memory bound or compute bound
        - compare the profiling results with roofline analysis to identify performance bottlenecks in the OpenCL kernel.
        - list top 1~2 bottlenecks kernel need to be optimized. 
    - Optimize the bottleneck OpenCL kernel to improve performance.
        - Abstract the required kernel and its input/output data format from openvino gpu primitive impl to generate a separated kernel file as the original kernels for kernel optimization, make sure the kernel optimization is implemented in a modular way to minimize the impact on other parts of the code.
        - Analyze required optimized kernel's input/output data format and configured parameters(layout, JIT, GWS/LWS, etc) from openvino gpu primitive impl based on E2E test, to construct the unit test for the optimized kernel so that can validate the optimized kernel's correctness and performance improvement through the unit test.
        - apply dev_kf_distill skill to optimize the kernel and unit test should run on remote machine, make sure the unit test of optimized kernel can be correctly executed and reach to 80% roofline performance.
    - Apply the optimized kernel into the GGUF support feature implementation and validate the performance improvement through remote build and E2E testing.
        - Get profiling data and compare the performance improvement after applying the optimized kernel into the GGUF support feature implementation, make sure the optimized kernel can be correctly executed and reach to 80% roofline performance based on the profiling results.

Key rules:
   - Don't push any commit to remote repo
   

Keep explanations conversational. For complex concepts, use multiple analogies.

---


