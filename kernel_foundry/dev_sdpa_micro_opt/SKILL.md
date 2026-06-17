---
name: dev_sdpa_micro_opt
description: Extract sdpa_micro kernels from openvino gpu plugin and generate standalone tests for performance optimization.
---

When extracting the sdpa_micro kernels from the OpenVINO GPU plugin, we generate a standalone test that can be built and run independently of the OpenVINO codebase. This allows us to focus on optimizing the performance of the sdpa_micro kernels without having to worry about the rest of the OpenVINO code.

0. **Top priority policies**
   - Don't create new commit, only modify code locally and remote copy to remote machine for build and test.
   - Don't push code to remote repository.
   - Follow the same code style and conventions used in the existing codebase to maintain consistency.
   - Don't change kernel input/output data's layout and format, only optimize the kernel code itself.

1. **Extract sdpa_micro kernels**: Use the existing code in the OpenVINO GPU plugin to extract the sdpa_micro kernels and create a standalone test that can be built and run independently. This involves copying the relevant kernel source files and any necessary headers, as well as creating a test executable that can run the kernels with various input parameters.
    - If have generated the separated kernel and standalone test, we can skip this step and directly use the generated test for optimization.
    - The original sdpa_micro kernels are located in `src/plugins/intel_gpu/src/graph/impls/ocl_v2`.
    - If possible, we can reference to `how_to_generate_separated_kernel.md` for the detailed steps to generate separated kernel and standalone test.

2. **Set up remote debugging environment**: Ensure that you have the necessary tools and access to the remote machine where test and validation can run. This may include setting up SSH access, installing debugging tools, and configuring the environment for remote debugging.
    - Remote machine information can be found in remote_machine.md
    - Setup some environment variables for remote debugging:
        export no_proxy=localhost,127.0.0.0/8,::1
        export ftp_proxy=http://child-prc.intel.com:913/
        export ftp_proxy=http://proxy-dmz.intel.com:912
        export https_proxy=http://proxy-dmz.intel.com:912
        export http_proxy=http://proxy-dmz.intel.com:912
        export HF_ENDPOINT=https://hf-mirror.com

3. **Run tests on remote machine**: Run test code on the remote machine and get the performance and analyze the roofline. 
    PA parameters for test:
        - tokens: 16
        - seqs: 1,2,3,4,5，6,7,8,9,10
        - history prompt size: 512
        - head-dim: 128
        - kv-heads: 8
        - heads: 32
    profilling tools in remote machine: cliloader
          cliloader install directory: C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe

4. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance to "SUMMARY.md"
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.


