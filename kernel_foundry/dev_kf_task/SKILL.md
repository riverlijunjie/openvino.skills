---
name: dev_create_kf_task
description: Create a Kernel Foundry task.
---

# Rules:
  - Don't create any commits
  - Don't push any commits
  - Don't create any pull requests

# Create a Kernel Foundry task
  To create a Kernel Foundry task, follow these steps:
  1. Must explicitly require the users to provide the kernel name, kernel language (e.g., OCL, CUDA), and the GPU architecture (e.g., bmg).
  2. Must explicitly require the users to provide the kernel task details, such as the operation to be performed, the input and output data formats, and any specific requirements or constraints for the kernel. 
  3. Must explicitly require the users whether they has a reference kernel, if yes, ask them to provide the kernel source code.
  4. Create a new directory for the task under the current directory with the given task name.
  5. Copy all files from `template` to the new task directory.
  6. Must update config.yaml with the provided kernel language and GPU architecture, also task_name and job_name:
      - task_name should be the same as the new directory name
      - job_name should be "test_" + task_name + "_" + kernel_language + "_" + gpu_architecture (e.g., test_gemm_ocl_bmg)
  6. Rename reference.cl to the kernel_name + "_reference.cl" (e.g., gemm_reference.cl) provided by user, and update the kernel source code to match the kernel task:
      - If user provides a reference kernel, modify the reference kernel according to the user's instructions.
      - If user doesn't provide a reference kernel, create a simple reference kernel that matches the specified operation and data formats.
      - Make sure the reference kernel is located between the tags `[REFERENCE_START]` and `[REFERENCE_END]` in the kernel source file.
      - Make sure the reference kernel can be compiled and run successfully, and produces correct results for the specified operation and data formats. Please write test samples to verify the reference kernel's correctness.
  7. Rename kernel.cl to the kernel_name + "_opt.cl" (e.g., gemm_opt.cl), and implement the new kernel task according to the user's instructions.
      - Must explicitly require the users if they want to provide more optimization ideas or directions for the new kernel task, and insert them between the tags `// [USER_INSTRUCTIONS_START]` and `// [USER_INSTRUCTIONS_END]` in the kernel source file.
      - Must explicitly require the users if they want to provide a base version of the optimized kernel, if yes, put them between the tags `// [EVOLVE_START]` and `// [EVOLVE_END]` in the kernel source file.
      - Make sure the optimized kernel is located between the tags `// [EVOLVE_START]` and `// [EVOLVE_END]` in the kernel source file.
      - If user provides a base version of the optimized kernel, make sure it can be compiled and run successfully, and produces correct results for the specified operation and data formats. Please write separated test samples to verify the optimized kernel's correctness.
  8. Update task.py to implement the testing logic for the new kernel task:
      - Due to kernel/task are decoupled, always auto-sync launch metadata to avoid manual drift.
      - Test harnesses should be robust to kernel evolution: prefer auto-derivation, and warnings.
      - Regression tests and path/import bootstrapping are essential for reliability.
      - Make sure task.py can compile and run both the reference kernel and the optimized kernel, and compare their results for correctness.
      - Make sure task.py can handle any specific requirements or constraints for the kernel, such as specific input/output data formats, or specific performance metrics.
      - Make sure task.py can be run with pytest and produces clear and informative output for the test results.
  9. After completing the above steps, run the tests locally to verify that everything is working correctly, and that the new kernel task is implemented correctly and efficiently.

  