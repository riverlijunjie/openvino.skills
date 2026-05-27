Linux:
      - machine: openvino-ci-74@odt-huyuan-openvino-ci-74
      - password: openvino
      - workspace: /mnt/river/kernel_foundry/workspace/copilot
      - Harware: B580(BMG), GPU frequency: 2900 MHz
          - f16 computation capability: 20*2048*2.9/1024=116.0 TFLOPS
          - memory bandwidth: 456 GB/s
          - XeCore count: 20
          - L3 cache: 32 MB
          - L2 cache: 128 KB/XeCore


Windows:
      - Target hardware: PTL
      - GPU frequency: 2400 MHz
      - Local_Admin@10.239.132.229
      - password:openvino
      - openvino directory: D:\river\moe\openvino
      - openvino.genai directory: D:\river\moe\openvino.genai
      - cliloader is loacted in: C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
      - TBB DLLs (needed for SSH sessions): D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
      - OV Runtime DLLs: D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
      - e2e test command:
           - cd C:\Users\Local_Admin
           - .\py310\Scripts\activate
           - cd D:\river\moe\openvino.genai\tools\llm_bench
           - run_moe.bat