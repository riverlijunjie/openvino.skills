---
name: dev_roofline_profiling
description: This skill is used to profiling roofline of specific platformand with specific models.
---

When profile roofline, first require to understand below prequistites:

  - Target Remote Linux machine: 
      - Target hardware: BMG
      - GPU frequency: 2850 MHz
      - openvino-ci-74@10.239.140.155
      - password:openvino
      - openvino directory: /mnt/river/model_loading/openvino
      - openvino.genai directory: /mnt/river/model_loading/openvino.genai
      - cliloader is loacted in: /mnt/river/model_loading/clintercept-3.0.6-Linux/bin
      - test utils directory: /mnt/river/model_loading/roofline_test_utils

  - Target Remote Windows machine with PTL 12Xe GPU: 
      - Target hardware: PTL
      - GPU frequency: 2400 MHz
      - Local_Admin@10.239.132.229
      - password:openvino
      - openvino directory: D:\river\moe\openvino
      - openvino.genai directory: D:\river\moe\openvino.genai
      - cliloader is loacted in: C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
      - test utils directory: D:\river\moe\dev_roofline_profiling\utils
      - TBB DLLs (needed for SSH sessions): D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
      - OV Runtime DLLs: D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
      - Results directory: D:\river\moe\roofline_results

  - Target Remote Linux machine with PTL 4Xe GPU: 
      - Target hardware: PTL
      - GPU frequency: 2400 MHz
      - intel@10.239.152.140
      - password:intel123
      - openvino directory:~/river/openvino
      - openvino.genai directory: ~/river/openvino.genai
      - cliloader is loacted in: ~/river/clintercept-3.0.6-Linux/bin/cliloader
      - test utils directory: ~/river/roofline_test_utils
      - TBB DLLs (needed for SSH sessions):~/river/openvino/temp/Linux_x86_64/tbb/lib
      - OV Runtime DLLs: ~/river/openvino/install_release/runtime/lib/intel64
      - Results directory: ~/river/roofline_results

  - Model: user should point out which model to use, if no assign then default is qwen3_moe
  - Input token size: user should provide token size, default values are 1024, 2048, 4096, 8192, 16K, 32K, 64K, 128K
  - Matmul Weights compression: 4-bit quantization with group size 128，and activation dtype is FP16，it will decompress weights to FP16 during computation.
  - LM_Head weights compression: 8-bit quantization with group size 128，and activation dtype is FP16，it will decompress weights to FP16 during computation.
  - SDPA should be converted to PA to get more accurate performance metrics, and the roofline will be based on PA's performance metrics.
  - MoE will use INT4 compression for expert weights and FP16 for activation, and all MoE computation will be fused into one pritimitive with 4~6 kernel, moe's mlp will be executed by fp16 precision.
  - When prefill, Matmul is computed by int8 XMX,  and the weights are stored and read in int4 but will be decompressed to int8 in memory. 
  - When decode, Matmul is computed by f16 XMX, and the weights are stored and read in int4 but will be decompressed to FP16 in memory.
  - KV cache is compressed by int8 quantization, and will be read as int8 and then decompressed to FP16 in memory, kv cache layout
        k: [num_blocks, num_kv_heads, head_size, block_size(16)]
        v: [num_blocks, num_kv_heads, block_size(16), head_size]
  - SDPA contains 3 implements, you need ask user to provide which one to be used, default is PagedAttention:
        - PagedAttention: it also has 2 different lauguage implement, one is opencl + micro_kernel, another is cm kernel. Default is opencl + micro_kernel implement, but if user want to use cm kernel implement, please make sure the target platform support cm kernel, Windows should supporte but Linux need add CM dependencies
        - SDPA kernels: it is opencl + micro_kernel implement
        - vlsdpa kernels: it cm kernel implement.
  - FC_Q, FC_K and FC_V will be fused into one gemm kernel: FC_QKV
  - If didn't find cliloder, you need download it from 

Then do proofline analysis with below steps:
1. **First top priority needed**:
    - Don't create new commit
    - Don't push code to remote repository
    - Don't submit PR
    - Don't change weights layout
2. **Read GPU specified architecture details**: Search for specific GPU architecture details to tailor optimizations effectively.
3. **Identify GPU architecture**: 
    - GPU architectures: Intel GPU LNL xe2 architecture.
        - Xe Core Number: 8
        - EU number of each Xe Core: 8
        - Threads number of each EU: 8
        - Subgroup size: 16 or 32
        - Register number of each EU: 256
        - Register size: 256 bytes
        - Shared Local memory size: 32KB
        - Video Memory Bandwidth: 100 GB/s
    - GPU architectures: Intel GPU B580(BMG) architecture.
        - Xe Core Number: 20
        - EU number of each Xe Core: 8
        - Threads number of each EU: 8
        - Subgroup size: 16 or 32
        - Register number of each EU: 256
        - Register size: 256 bytes
        - Shared Local memory size: 32KB
        - Video Memory Bandwidth: 456 GB/s
    - GPU architectures: Intel PTL 12Xe GPU architecture
        - Xe Core Number: 12
        - EU number of each Xe Core: 8
        - Threads number of each EU: 10
        - Subgroup size: 16 or 32
        - Register number of each EU: 256
        - Register size: 256 bytes
        - Shared Local memory size: 32KB
        - Video Memory Bandwidth: 110 GB/s
        - GPU frequency: 2450 MHz
    - GPU architectures: Intel PTL 4Xe GPU architecture
        - Xe Core Number: 4
        - EU number of each Xe Core: 8
        - Threads number of each EU: 10
        - Subgroup size: 16 or 32
        - Register number of each EU: 256
        - Register size: 256 bytes
        - Shared Local memory size: 32KB
        - Video Memory Bandwidth: 110 GB/s
        - GPU frequency: 2450 MHz
    - GPU architectures: Intel(R) Arc(TM) B390 GPU (96CUs, 2400MHz)
        - It is same with Intel PTL GPU architecture due to recongnition issue of cliloader, so we will use the same performance metrics for both PTL and B390, and also use the same theoretical roofline for both PTL and B390.
4. **Probe hardware capabilities**: 
    - Use tools like clinfo, clpeak, cliloader to probe the hardware capabilities and collect performance metrics for the target GPU architecture.
       - clpeak may provide memory bandwidth with cache hit, we should excluse such data.
       - clinfo can provide some basic GPU information, such as GPU frequency, EU number, hardware threads number, subgroup size, shared local memory size.
    - Write our own test utils based on opencl to collect the performance metrics for the target GPU architecture, such as FLOPS, memory bandwidth, EU number hardware threads, EU number, subgroup size, shared local memory size, etc， then to compare with above tools results to make sure the performance metrics are accurate.
      - Write a test util to measure the actual memory bandwidth of the target GPU architecture by performing a large memory copy operation and measuring the time taken, which can provide insights into the effective memory bandwidth that can be achieved on the target GPU architecture for real workloads.
      - Write a test util based on opencl to get gpu info, such as GPU frequnecy, EU number, hardware threads number, subgroup size, shared local memory size, etc.
    - Hardware capabilities can also be obtained from Intel GPU architecture specification documents, such as the Intel GPU architecture whitepapers, technical reference manuals, and other official documentation provided by Intel.

5. **Compute hardware capabilities**:
    - Compute the theoretical roofline for the target GPU architecture based on its specifications (e.g., memory bandwidth, compute throughput).
        - Computation throughput can be calculated based on the number of EUs, threads, and the clock frequency. For example, for BMG:
            - FP16 XMX peak: 20 (Xe cores) × 8 (EUs/core) × 256 (FLOPs per cycle for FP16 XMX ) × 2850 MHz = 116.736 TFLOPS
            - INT8 XMX peak: 20 (Xe cores) × 8 (EUs/core) × 8 (threads/EU) × 4 (FLOPs per cycle for INT8) × 2850 MHz = 233.472 TOPS
    - Use the roofline model to identify whether the kernel is memory-bound or compute-bound, and to set performance targets for optimization.

6. **Analyze model architecture and map to OpenVINO ops**:
    - Get the model architecture from:
         local: ~/workspace/transformers/src/transformers/models
         or remote: https://github.com/huggingface/transformers/blob/main/src/transformers/models
    - Analyze the model architecture to understand its computational patterns and memory access patterns.
    - Identify potential bottlenecks in the model that could impact performance on the target GPU architecture.
    - Map the model operations to OpenVINO operations to leverage hardware acceleration and optimize performance.
        - Identify which parts of the model can be accelerated using OpenVINO's optimized kernels, such as MatMul, PagedAttention, etc., and which parts may require custom implementations or optimizations.
        - Figure out how many calls for each OpenVINO operation during inference, and also figure out the input and output tensor shapes for each OpenVINO operation, which can help us to analyze the computational intensity and memory access patterns of each operation, and also can help us to design the test utils for performance metrics collection. Notice for the same input token size of model, the different input/outputtensor shape will be looked as different ops, because the performance metrics of the same op with different input/output tensor shape can be very different, and we should analyze them separately to get more accurate performance metrics and also can provide more insights for optimization.
        - Try to fuse operations where possible to reduce memory bandwidth requirements and improve compute efficiency.
        - Some eltwise can be fused into MatMul or PA to reduce memory access and improve performance, such as bias add, gelu, etc. We can analyze the model architecture to find out those eltwise and fuse them into MatMul or PA if possible.
        - MoE will use INT4 compression for expert weights and FP16 for activation, and all MoE computation will be fused into one pritimitive with 4~6 kernel, moe's mlp will be executed by fp16 precision.
        - List each OpenVINO operation used in the model and each operation's parameters (e.g., input/output shapes, data types) to understand their computational requirements and how they map to the GPU's capabilities.
        - For each OpenVINO operation, analyze its computational intensity (FLOPs) and memory access patterns to determine whether it is likely to be memory-bound or compute-bound on the target GPU architecture.
        - Save or update the operation list and their parameters in a structured format (e.g., JSON, CSV) for reference during optimization and performance analysis, file name is ops_mapping.json.
     - Provide roofline analysis for each OpenVINO operation, comparing its computational intensity and memory access patterns against the theoretical roofline of the target GPU architecture to identify potential performance bottlenecks and optimization opportunities.
     - Use each operation's roofline analysis and calling counts to figure out the model's overall hardware roofline:
        - prefill roofline: all input token size
        - decode roofline: all input token size, and only consider the 2nd token generation

7. **Profile the model's actual performance roofline**:
    - Read ops_mapping.json and get the ops name and its parameters(input and output tensor shape, data type, calling, countetc), and then write test utils to collect performance metrics for each OpenVINO operation used in the model on the target GPU architecture.
    - If don't have any ops test case in test utils, you should write/update utils based on OpenVINO API to test each OpenVINO operation's performance metrics
    - Test utils should be designed to run on the target remote machine and collect accurate performance metrics for each OpenVINO operation used in the model, and also can avoid the impact of other ops on performance metrics collection. Below are some guidelines for writing test utils:
        - Test utils should be written by C++ and reference to OpenVINO unit test code for correct usage of OpenVINO API and to make sure the test case can run successfully in remote machine, and also can provide accurate performance metrics.
        - For testing SDPA performance, there will be several choose for difference scenarios, user should choose the most suitable one for their model:
            - PagedAttention: we can reference to src/plugins/intel_gpu/tests/unit/test_cases/paged_attention_gpu_test.cpp to write the test utils for PagedAttention performance metrics collection. Dont's use gemm+softmax+gemm to test PA performance, because it will cause extra memory access and computation that is not part of the actual PA kernel, and it will be hard to isolate the performance metrics of PA kernel itself. Instead, we can directly use OpenVINO's built-in PagedAttention op to test the performance of PA kernel, which can provide more accurate performance metrics that reflect the true computational and memory access patterns of the PA kernel without being skewed by other ops.
            - SDPA performance: we can reference to src/plugins/intel_gpu/tests/unit/test_cases/sdpa_gpu_test.cpp to write the test utils for SDPA performance metrics collection, and also we should make sure the test case can reflect the actual SDPA performance by using the same weights compression and data type as actual SDPA in the model, and also make sure the input and output tensor shape is same as actual SDPA in the model, which can provide more accurate performance metrics that reflect the true computational and memory access patterns of SDPA in the model.
            - vlsdpa performance: firstly we should confirm that target platform support CM kernel, Windows should supporte but Linux need add CM dependencies; we can reference to src/plugins/intel_gpu/tests/unit/test_cases/vlsdpa_gpu_test.cpp to write the test utils for vlsdpa performance metrics collection, and also make sure the test case can reflect the actual vlsdpa performance by using the same weights compression and data type as actual vlsdpa in the model, and also make sure the input and output tensor shape is same as actual vlsdpa in the model, which can provide more accurate performance metrics that reflect the true computational and memory access patterns of vlsdpa in the model.
        - For testing MatMul performance, we can reference to src/plugins/intel_gpu/tests/unit/test_cases/fully_connected_gpu_test.cpp to write the test utils for MatMul performance metrics collection.
        - For MoE performance, you can reference below files to write the test utils for MoE performance metrics collection, but you need to modify the test case to make it can reflect the actual MoE performance by using the same weights compression and data type as actual MoE in the model, and also make sure the input and output tensor shape is same as actual MoE in the model, which can provide more accurate performance metrics that reflect the true computational and memory access patterns of MoE in the model.
            - src/plugins/intel_gpu/tests/unit/transformations/convert_moe_to_compressed_test.cpp
            - src/plugins/intel_gpu/tests/unit/transformations/fuse_moe_3gemm_compressed_test.cpp
            - src/plugins/intel_gpu/tests/unit/test_cases/moe_3gemm_gpu_test.cpp
        - MoE contains 4~6 kernels(contains grouped_micro_gemm(gate/up/down) and moe_* kernel), we should analyze each kernel's performance metrics separately, and also analyze the overall MoE performance by aggregating the performance metrics of each kernel, which can provide more insights for optimization. For example, if we find that the performance of the MoE is bottlenecked by the memory access of the weights, we can try to optimize the weights layout or compression method to improve the performance.
        - MoE shared expert will be fused to MoE primitive, reference moe_3gemm_compressed_gpu_shared_random of /home/ov2022/workspace/remote_debug/openvino/src/plugins/intel_gpu/tests/unit/test_cases/moe_3gemm_gpu_test.cpp for shared expert test case, and we can also design our own test case based on that to make it more close to the actual shared expert in the model, such as using the same input and output tensor shape as actual shared expert in the model, and also using the same weights compression and data type as actual shared expert in the model, which can provide more accurate performance metrics for shared expert and also can help us to optimize shared expert based on the actual performance characteristics on the target GPU architecture.
        - For other ops performance testing, we can reference to src/plugins/intel_gpu/tests/unit/test_cases/ to find the relevant test cases for other ops, and write the test utils based on those test cases.
        - Test utils will be put into utils folder and each ops should have its own subdirectory and test app, so that we can use cliloader to profiling each ops performance metrics independently, and also can avoid the impact of other ops on performance metrics collection.
        - swish and multiply should be fused into corresponding kernels, and don't profile it separately    
        - It is better to use cliloader to run the test utils for each ops, we should run test app for each input token and use cliloader to get each kernels average performance data and use a python tool to parse the cliloader log and get the performance metrics, such as latency. Don't put all input token size performance metrics collection in one test app, because it will cause the performance metrics of different input token size get mixed together and hard to analyze. Instead, we can run the test app for each input token size independently, and use cliloader to get the performance metrics for each input token size separately, which can provide more accurate performance metrics for each input token size and also can make it easier to analyze the impact of input token size on performance.
        - lm_head should be tested separately from matmul, because it has different weights compression and data type, and it will have different performance characteristics on the target GPU architecture, and also it will be easier to analyze the performance metrics of lm_head separately without being skewed by the performance metrics of matmul. lm_head only is used for single token, so just test lm_head for single token and used for prefill and decode.
        - Each ops test case should be designed to avoid L3 cache reuse as much as possible, to get more accurate performance metrics that reflect the true computational and memory access patterns of the operation without being skewed by cache effects. A simple way is to create multiple input and output tensors, and make sure each test iteration uses different input and output tensors to avoid reuse L3 cache. 
        - For dGPU we should make input tensor to allocate usm_device memory to avoid the impact of PCIe transfer on performance metrics collection, and also can get more accurate performance metrics that reflect the true computational and memory access patterns of the operation on dGPU. For integrated GPU, we can use usm_host or usm_device memory for input and output tensor, because integrated GPU
        - All the performance metrics will be saved in performance_metrics.json
    - Remote copy ops_mapping.json and test utils to openvino directory of target remote machine, then build and run the test utils to collect performance metrics for each OpenVINO operation used in the model on the target GPU architecture.
    - Use test utils to collect each ops performance metrics with help of cliloader, and save the metrics in performance_metrics.json
        - Set enough iterations for each ops test app to get stable performance metrics, which will reduce the impact of warm-up time and other transient effects on the collected performance metrics, and can provide a more accurate representation of the operation's performance characteristics on the target GPU architecture.
    - Based on the collected performance metrics and each ops calling time in inference, analyze the model's actual performance roofline
        - We can igore tiny ops that have very small computational intensity or small memory access, and focus on the ops that have significant impact on the model's overall performance roofline
        - For each significant op, compare its actual performance metrics against the theoretical roofline to identify whether it is performing as expected or if there are bottlenecks that need to be addressed.
        - Analyze the overall model's roofline by aggregating the performance metrics of the significant ops and comparing it against the theoretical roofline to understand the model's performance characteristics on the target GPU architecture. We aslo need add each ops calling time percentage in overall inference to the roofline analysis, which can help us to identify which ops are the main bottleneck for the model's performance and also can provide insights for optimization.

8. **Some important tips need to considered**
    - Remove swiglu_ref from small ops benchmarks and data collection.
    - Use average (mean) kernel time instead of min from cliloader
    - Calculate efficiency percentage for each op relative to hardware capability (BW% for memory-bound, XMX% for compute-bound)
    - Math computaion, exp is 30 fops, sin and cos is 10 fops, sqrt is 10 fops, and use these numbers to calculate the theoretical roofline for those ops, instead of using 1 fop for each element.
    - prefill will use dynamic_quantize_gpu_opt kernel, please also analyze its performance metrics as other kernels
    - pa kernel should be split to kv_cache_update and pa computation 2 parts to profiling
    - PA prefill uses causal mask (lower-triangular attention), so the actual computation is only half of the full attention matrix. When calculating theoretical FLOPs for PA prefill, use effective attention pairs = Sq*(Sq+1)/2 ≈ Sq²/2 instead of Sq*Skv. For decode (Sq=1), no causal mask reduction applies since the single query attends to all past KV tokens.
    - Should list each op's performance metrics: average latency, total calls time per inference, total latency, achieved FLOPS, achieved memory bandwidth, and efficiency percentage relative to the theoretical roofline.
    - ops test iterations should be set to a proper value, should make this ops test can last more than 1000ms total kernel execution time, to get stable performance metrics, a simple moethod is 1000 ms divide the theorial latency of each op, and set the iterations to be a bit more than that to make sure the total kernel execution time can be more than 1000ms.
    - Change all time units to ms
    - If some ops in this models have different parameters with current test utils, we should adjust and update test app and test scripts to match it.
    - If new models contains new ops that not in current test utils, we should write new test utils to collect performance metrics for those new ops, and also add those new ops into ops_mapping.json, and then run the new test utils to get the performance metrics for those new ops.
    - If new models contains new ops that is not supported by current OpenVINO, we should use theretical performance metrics based on the GPU architecture and the computational requirements of those new ops to estimate their performance characteristics and how they may impact the overall roofline of the model, and also can provide insights for optimization. We can also try to implement those new ops by using OpenVINO's custom kernel feature, and then write test utils to collect performance metrics for those new ops, which can provide more accurate performance metrics.
    - If some ops exceed 100% efficiency, we should check the performance metrics collection method and test utils design to make sure the performance metrics are accurate, and also check if there is any optimization in OpenVINO that can cause the performance metrics exceed the theoretical roofline, such as kernel fusion, etc.
    - Test scripts should be copied to test utils directory of remote machine and run in remote machine 
    - Save all cliloader perf logs to a new logs subdirectory in the test directory
    - All files related to roofline analysis should be organized in a new created folder with clear structure, such as:
        - utils/: all test utils and scripts for performance metrics collection
        - db/: all scripts and files related to performance metrics database, such as build_db.py and metrics.db
        - logs/: all cliloader perf logs
        - outputs/:
              - performance_metrics.json: all collected performance metrics for each op
              - SUMMARY files: the summary of roofline analysis, findings, and insights.
    - Dont write file to /temp directory, but to outputs directory
    - Sleep time doesn't exceed 30s between test cases

9. **Document optimization**:
    - Summarize the roofline analysis process, findings, and insights in a clear and concise manner.
    - Summarize the performance characteristics of each significant op and the overall model roofline: single latency, calling counts, total latency, achieved FLOPS, achieved memory bandwidth, and efficiency percentage relative to the theoretical roofline, memory bound ot compute bound; It should contains prefill(one token size one table) and decode(one token size one table) performance metrics and roofline analysis separately.
    - Summarize the total latency of the model, include prefill and decode, and also summarize the latency percentage of each significant op in overall inference
    - All analysis and findings should be documented in "SUMMARY_<model_name>_<date>.md" with clear explanations of the roofline analysis results, identified bottlenecks, and potential optimization opportunities, and all material should be input a new created folder
    - Rewrite "SUMMARY_<model_name>_<date>.md" with the new roofline analysis methods. Add the new performance metrics collected for each OpenVINO operation, and update the overall model roofline analysis based on the new performance metrics. Make sure to explain any changes in the model's performance characteristics and how they relate to the theoretical roofline of the target GPU architecture.
    - SUMMARY should do performance anaylsis and comparsion for prefill and decode separately; For example, prefill may be more memory-bound due to larger input token size, while decode may be more compute-bound due to smaller input token size and more complex computation patterns.
    - SUMMARY should contains measured performance metrics for each significant op, such as latency, achieved FLOPS, achieved memory bandwidth, and efficiency percentage relative to the theoretical roofline, and also should contain the overall model roofline analysis based on the aggregated performance metrics of the significant ops, and also should provide insights for optimization based on the roofline analysis results.
    - SUMMARY should contains series of table to show the performance metrics for each significant kernels, such as single latency, calling times, total latency, achieved FLOPS, achieved memory bandwidth, and efficiency percentage relative to the theoretical roofline.
    - SUMMARY should contain a clean kernel breakdown
    - SUMMARY should contain series of table to show: ops name, kernel name, single latency, calling times, total latency, achieved FLOPS, achieved memory bandwidth, and efficiency percentage relative to the theoretical roofline, memory bound or compute bound, for each significant op in prefill and decode separately, and one table list one sequence of input token size for prefill or decode. For example, theret are 1024, 2048, 4096, 8192, 16K, 32K, 64K, 128K input token size for prefill, then we should have 8 tables for prefill performance metrics and roofline analysis, and each table is for one input token size. And also we should have 8 tables for decode performance metrics and roofline analysis, and each table is for one input token size.
    - The SUMMARY contains 8 decode + 8 prefill per-kernel tables (op / kernel / single ms / calls / total ms / GFLOPS / GB/s / Eff% / bound), bottleneck breakdowns, top optimization levers, reproduction commands, and caveats
    - Update all scripts for roofline analysis in the "utils" folder, and document how to use them for future reference.
    - Put all logs data into a database or structured format for easy querying and analysis in the future, and also can help us to track the performance metrics over time and across different models and GPU architectures.
    - Remove data out of date, such as old performance metrics, and make sure all the information in "SUMMARY_<model_name>_<date>.md" is accurate and up to date.

Reference skills: intel-gpu-hw-info

Keep explanations conversational. For complex concepts, use multiple analogies.