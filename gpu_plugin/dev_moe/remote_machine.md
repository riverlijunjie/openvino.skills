Linux:
      - machine: openvino-ci-74@odt-huyuan-openvino-ci-74
      - password: openvino
      - openvino path: /mnt/river/model_loading/openvino
      - test command: ./ov_gpu_unit_tests --gtest_filter=*moe_3gemm*mix*


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
           - cd D:\river\moe\openvino.genai\samples\python\text_generation
           - run_qwen3.5.bat