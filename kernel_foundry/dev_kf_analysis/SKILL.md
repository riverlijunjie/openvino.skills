---
name: dev_kf_analysis
description: Analyze kf framework and its components, figure out the gaps and areas for improvement.
---

# Rules:
  - Don't create any commits
  - Don't push any commits
  - Don't create any pull requests

# Understand Kernel Foundry framework and analyze its components
  To understand the Kernel Foundry framework and analyze its components, follow these steps:
    1. Read the source code:
          - /mnt/river/kernel_foundry/kernelfoundry.internal
          - /mnt/river/kernel_foundry/kernelfoundry.kernel-eval
          - /mnt/river/kernel_foundry/kernelfoundry.templates
    2. Study the overall architecture of the Kernel Foundry framework, including its main components such as the kernel generator, the search algorithm, the performance model, and the testing framework.
    3. Analyze how these components interact with each other during the kernel optimization process, and identify any potential bottlenecks or areas for improvement.
    4. Review the existing documentation and codebase of the Kernel Foundry framework to gain a deeper understanding of its implementation details and design choices.
    5. Identify any gaps in the current framework, such as missing features, limitations in the search algorithm, or inefficiencies in the performance model.
    6. Propose potential improvements or enhancements to address these gaps, such as new optimization strategies, more efficient search algorithms, or better performance modeling techniques.
    7. Document your analysis and proposed improvements in a clear and concise manner, providing evidence and reasoning to support your conclusions.
    8. Document will be named as `KF_ANALYSIS.md` and placed in the root directory of this skill.


# Understand kernel-design-agents framework and analyze its components
    To understand the kernel-design-agents framework and analyze its components, follow these steps:
    1. Read the source code /mnt/river/kernel_foundry/kda/kernel-design-agents
    2. Study the overall architecture of the kernel-design-agents framework, including its main components such as the agent design, the environment setup, the reward system, and the training process.
    3. Analyze how these components interact with each other during the agent's learning process, and identify any potential bottlenecks or areas for improvement.
    4. Review the existing documentation and codebase of the kernel-design-agents framework to gain a deeper understanding of its implementation details and design choices.
    5. Identify any gaps in the current framework, such as limitations in the agent's learning capabilities, inefficiencies in the environment setup, or weaknesses in the reward system.
    6. Propose potential improvements or enhancements to address these gaps, such as new learning algorithms, more efficient environment setups, or better reward systems.
    7. Document your analysis and proposed improvements in a clear and concise manner, providing evidence and reasoning to support your conclusions.
    8. Document will be named as `KDA_ANALYSIS.md` and placed in the root directory of this skill.


# Design one documentation for kernel optimization agent framework
    1. Reuse the existing kernel-design-agents framework as much as possible, and make necessary modifications to fit the specific requirements of kernel optimization.
    2. Elimate negative impact of original design of kernel_foundry
        - Huge token consumption for each trial (e.g., 50K tokens per trial) → design more efficient prompt templates, and optimize the interaction between the agent and the environment to reduce unnecessary token usage.
        - Long evaluation time for each trial (e.g., 1-2 minutes per trial) → optimize the testing framework to speed up the evaluation process, such as using more efficient test cases, parallelizing the testing process, or using performance modeling to predict the performance of the optimized kernel without running the full tests.
        - MAP-ELites algorithm may converge slowly and get stuck in local optima → design more effective search algorithms, such as using reinforcement learning techniques, error/regression fallback policy, evolutionary strategies, or other optimization algorithms that can explore the search space more efficiently and effectively.
        - Too complex task configuration and setup process → simplify the task configuration and setup process, such as providing clear documentation, automating the setup process, or providing user-friendly interfaces for configuring and running the optimization tasks.
        - Test case for evolved kernel is static and can changing with kernels evolution → design more comprehensive and robust test cases for the evolved kernels, such as automated jit and gws/lws matching, covering a wider range of input data, jit testing for edge cases, and ensuring that the test cases can effectively evaluate the performance and correctness of the optimized kernels.
    3. Agent requirements:
        - The agent should use a natural language requirement to trigger the optimization process and interact with the environment, allowing users to provide instructions, feedback, and constraints in a human-readable format.
        - The agent should be able to learn from the feedback and rewards provided by the environment, and use this information to improve its optimization strategies over time.
        - The agent should be able to generate optimized kernel code based on the requirements and constraints provided by the user, and ensure that the generated code is correct and efficient.
        - The agent should be able to handle a wide range of optimization tasks, such as optimizing for different performance metrics (e.g., latency, throughput, energy efficiency), optimizing for different hardware architectures, and optimizing for different types of kernels (e.g., matrix multiplication, convolution, etc.).
        - The agent should be able to provide clear and informative feedback to the user about the optimization process, such as the performance improvements achieved, the changes made to the kernel code, and any potential trade-offs or limitations of the optimized kernel
        - The agent should be able to end-to-end closed loop optimization, including requirement understanding, kernel generation, testing, and learning from feedback, without requiring manual intervention in the loop.
        - The agent should be designed to be extensible and adaptable, allowing for easy integration of new optimization techniques, new hardware architectures, and new types of kernels as needed in the future.
        - The agent should be designed to be efficient and scalable, able to handle large search spaces and complex optimization tasks without excessive computational resources or time.
        - The agent can reuse the existing components of kernel_foundry, such as the testing framework, optimization algorithms, and kernel generation tools, to leverage the existing infrastructure and reduce development effort.
        - The agent should avoid reward hacking or overfitting to specific test cases, and instead focus on general optimization strategies that can improve performance across a wide range of scenarios and inputs.
        - The agent should provide interface to do e2e testing and evaluation of the optimized kernels, ensuring that the optimized kernels can be effectively evaluated for both performance and correctness in a comprehensive manner.
        - The agent should contains series of assets, such as tools, skills and database for good seeds.
    4. Can reference to a draft design document for kernel optimization agent framework: `reference.md`
    5. Document the design of the kernel optimization agent framework
        - Document in a clear and concise manner, providing detailed explanations of the architecture, components, interactions, and optimization strategies used by the agent.
        - Include examples and use cases to illustrate how the agent can be used for different optimization tasks and scenarios. The documentation should be organized in a logical manner, with clear sections and headings to facilitate easy navigation and understanding for users and developers who want to use or contribute to the kernel optimization agent framework.
        - Document will be named as `DESIGNED.md` and placed in the root directory of this skill.
