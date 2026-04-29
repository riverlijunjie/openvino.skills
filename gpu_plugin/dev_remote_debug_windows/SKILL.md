---
name: dev_remote_debug_windows
description: Remote debug kinds of issues of gpu plugin inference on Windows.
---

When remote debugging kinds of issues of GPU plugin inference, the following steps can be taken to:

1. **Top priority policies**
   - Don't create new commit, only modify code locally and remote copy to remote machine for build and test.
   - Don't push code to remote repository.
   - Follow the same code style and conventions used in the existing codebase to maintain consistency.

2. **Set up remote debugging environment**: Ensure that you have the necessary tools and access to the remote machine where the GPU plugin inference can run. This may include setting up SSH access, installing debugging tools, and configuring the environment for remote debugging.
    - Remote machine information can be found in remote_machine.md
    - Setup some environment variables for remote debugging:
        set no_proxy=localhost,127.0.0.0/8,::1
        set ftp_proxy=http://child-prc.intel.com:913/
        set https_proxy=http://proxy-dmz.intel.com:912
        set http_proxy=http://proxy-dmz.intel.com:911
        set HF_ENDPOINT=https://hf-mirror.com
    - Check if OpenVINO is already downloaded in remote machine: target directory can be found in remote_machine.md
    - If OpenVINO has not been downloaded, download OpenVINO to remote machine
        command: git clone https://github.com/openvinotoolkit/openvino.git
        remote copy build_openvino.bat to remote machine and put into openvino directory
    - Build OpenVINO in remote machine:
        - cd openvino
        - build_openvino.bat
    - Download openvino.genai if not exist in remote machine:
        command: git clone https://github.com/openvinotoolkit/openvino.genai.git
        remote copy build_openvino_genai.bat to remote machine and put into openvino.genai directory
        remote copy test script run_llm.bat to remote machine and put into openvino.genai/tool/ directory
        setup python environment for openvino.genai in remote machine:
        - cd openvino.genai
        - python3 -m venv venv
        - venv\Scripts\activate
        - cd tools\llm_bench
        - pip install -r requirements.txt
    - Build openvino.genai in remote machine:
        - cd openvino.genai
        - build_openvino_genai.bat

3. **Reproduce the issue and get logs**: Run the GPU plugin inference code on the remote machine and verify that the issue can be reproduced. This will help you understand the extent of the issue and provide a baseline for measuring improvements.
    - Remote copy modified code to remote machine for build and test.
    - Remote build openvino, fllow the build configure of build_openvino.bat
    - setup python venv and activate before running the test script: %python_venv_dir%\Scripts\activate
    - cd openvino.genai\tools\llm_bench
    - run the test script: run_llm.bat
    - analyze the logs to get required info.

4. **If needed to profile the code**: Use performance profiling tools to analyze the GPU plugin inference code and identify bottlenecks or areas that are causing performance regressions.
    - Performance profiling tools in remote machine: vtune

5. **Identify optimization opportunities**: Look for areas in the code where performance can be improved, such as inefficient algorithms, redundant computations, or memory bottlenecks.
6. **Implement optimizations**: Apply code changes to optimize the identified areas. This may include algorithmic improvements, better memory management, or leveraging hardware capabilities.
    - Follow the same code style and conventions used in the existing codebase to maintain consistency.
    - Don't create new commit
    - Don't push code to remote repository
    - Modify code locally and remote copy to remote machine for build and test.
7. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance to "SUMMARY.md"
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.


