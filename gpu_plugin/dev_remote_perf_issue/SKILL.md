---
name: dev_remote_perf_issue
description: Remote debug performance regression issues of gpu plugin inference.
---

When remote debugging performance regression issues of GPU plugin inference, the following steps can be taken to identify and optimize the code:

1. **Top priority policies**
   - Don't create new commit, only modify code locally and remote copy to remote machine for build and test.
   - Don't push code to remote repository.
   - Follow the same code style and conventions used in the existing codebase to maintain consistency.

2. **Set up remote debugging environment**: Ensure that you have the necessary tools and access to the remote machine where the GPU plugin inference can run. This may include setting up SSH access, installing debugging tools, and configuring the environment for remote debugging.
    - Remote machine: gta@10.80.67.42, pwd:gta
    - Setup some environment variables for remote debugging:
        export no_proxy=localhost,127.0.0.0/8,::1
        export ftp_proxy=http://child-prc.intel.com:913/
        export ftp_proxy=http://proxy-dmz.intel.com:912
        export https_proxy=http://proxy-dmz.intel.com:912
        export http_proxy=http://proxy-dmz.intel.com:912
        export HF_ENDPOINT=https://hf-mirror.com
    - Check if OpenVINO is already downloaded in remote machine:
        target directory: /home/gta/openvino
    - If OpenVINO has not been downloaded, download OpenVINO to remote machine
        command: git clone https://github.com/openvinotoolkit/openvino.git
        remote copy build_openvino.sh to remote machine and put into openvino directory
    - Build OpenVINO in remote machine:
        - cd openvino
        - source build_openvino.sh
    - Download openvino.genai if not exist in remote machine:
        command: git clone https://github.com/openvinotoolkit/openvino.genai.git
        remote copy build_openvino_genai.sh to remote machine and put into openvino.genai directory
        remote copy test script run_llm.sh to remote machine and put into openvino.genai/tool/ directory
        setup python environment for openvino.genai in remote machine:
        - cd openvino.genai
        - python3 -m venv venv
        - source venv/bin/activate
        - cd tools/llm_bench
        - pip install -r requirements.txt
    - Build openvino.genai in remote machine:
        - cd openvino.genai
        - source build_openvino_genai.sh

3. **Reproduce the performance regression**: Run the GPU plugin inference code on the remote machine and verify that the performance regression can be reproduced. This will help you understand the extent of the issue and provide a baseline for measuring improvements.
    - cd openvino.genai
    - source venv/bin/activate
    - cd tools/llm_bench
    - run the test script: bash run_llm.sh
    - get prefill and decode latency from the test script output, and compare with the historical performance data to confirm the performance regression issue.

4. **If needed to profile the code**: Use performance profiling tools to analyze the GPU plugin inference code and identify bottlenecks or areas that are causing performance regressions.
    - Performance profiling tools in remote machine: vtune
          vtune install directory: /opt/intel/oneapi/vtune/latest/bin64/vtune
          command to run vtune: /opt/intel/oneapi/vtune/latest/bin64/vtune -collect hotspots -result-dir vtune_result -- <command_to_test_script>

2. **Identify optimization opportunities**: Look for areas in the code where performance can be improved, such as inefficient algorithms, redundant computations, or memory bottlenecks.
3. **Implement optimizations**: Apply code changes to optimize the identified areas. This may include algorithmic improvements, better memory management, or leveraging hardware capabilities.
    - Follow the same code style and conventions used in the existing codebase to maintain consistency.
    - Don't create new commit
    - Don't push code to remote repository
    - Modify code locally and remote copy to remote machine for build and test.
4. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance to "SUMMARY.md"
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.


