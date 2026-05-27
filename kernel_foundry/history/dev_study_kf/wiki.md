
KernelFoundry

Overview

KernelFoundry is an AI-powered kernel code-generation platform developed by Intel's Client AI Solutions & Planning team that leverages Large Language Models (LLMs) to automatically generate, optimize, and correct computational kernels using an evolutionary algorithm.

Core Capability

The platform automates the traditionally manual and time-intensive process of writing high-performance computational kernels by:

Accepting task definitions and test specifications from developers
Employing an evolutionary LLM algorithm that iteratively generates, debugs, and optimizes kernel code
Delivering production-ready kernels through automated refinement cycles


Key Features
Dual Access Methods: Web-based UI and VS Code extension for flexible workflow integration
Template-Based Development: Pre-built templates for common tasks (e.g., PyTorch-based tasks) accelerate time-to-value
Hardware Agnostic: Supports both Intel (XPU SYCL) and NVIDIA (CUDA) hardware platforms
Minimal Dependencies: Lightweight core package (Python 3.12) with modular testing requirements
Business Value
Accelerated Development: Reduces kernel development time from days/weeks to hours through automated generation
Quality Assurance: Built-in testing framework ensures generated kernels meet functional requirements
Accessibility: Democratizes kernel optimization by reducing the specialized expertise barrier
Iterative Optimization: Evolutionary approach continuously improves performance without manual intervention
Target Users

Performance engineers, ML researchers, and software developers requiring optimized computational kernels for AI/ML workloads, scientific computing, or high-performance applications.

Technical Foundation

Built on Python 3.12 with PyTorch integration, leveraging state-of-the-art LLMs to understand computational requirements and generate hardware-optimized code.


Created by Wiedemann, Nina, last updated by Ummenhofer, Benjamin on May 13, 2026  6 minute read
The config.yaml  file for a custom task sets the parameters for KernelFoundry's algorithm. This page provides an overview of the important parameters and their functionality. 

2a.1 Quick Start:
To create a config file for your needs, try our configuration tool at http://codegen-head1.imu.intel.com:8889/job/submission ("Config generator").
For examples, see our config templates: https://github.com/intel-sandbox/kernelfoundry.templates/tree/main/config_templates 
A comprehensive list of parameters and explanations is provided below.

2a.2 Required Parameters
As you can see in the provided templates, certain parameters must be defined for each custom task: 

task_name
job_name
language
gpu_arch
See the table below for explanations.

2a.3 Important Top-Level Parameters
task_name	string	A name for the type of operation that the kernel should implement (choose any, e.g. "relu")	
job_name	string	A name for the "job," which is one execution of KernelFoundry's algorithm
(iterative generation and optimization of the kernel)	
language	string	Programming language of the kernel. Important: this will determine which compiler and profiler are used.
Currently supported: "SYCL", "CUDA", "triton", "OCL"	
gpu_arch	string	GPU architecture to test the kernel on. Currently supported: "ptl", "lnl", "bmg", "Ampere".
Note: Coupled with language, e.g., "Ampere" can only be used in combination with "CUDA".	
max_iters	int	Maximum iterations to run 	3
branches_per_iteration	int	How many kernels to generate in each iteration (calls to LLM)	1
evolve_mode	bool	
Whether to use evolutionary algorithm
(otherwise it just takes the kernel from the previous iteration and tries to improve it)

false
stop_once_correct	bool	Whether to stop once a correct kernel has been found (set true for translation)	false
build_timeout	int	Timeout for building (compilation), in seconds (increase if you see build timeout errors)	200
test_timeout	int	Timeout for kernel execution, including correctness testing, benchmarking and profiling, in seconds. 
Increase if you experience test timeouts (e.g. if tests require model or data loading).	
300

test_reference	bool	
Whether to test the reference implementation (correctness test and benchmarking).
Should usually be true to allow computing the speedup, but can be set false e.g. if the reference is
only used for the prompt but is not executable.

true

has_build_step	bool	
Whether the task has a build step (build function in test class). Usually true to compile the kernel.

true

has_reference_build_step	bool	
Whether the reference implementation has a build step (build_reference function in test class).

Can be false e.g. if the reference is Pytorch code. Defaults to dummy build if true but no build function found.

true

use_feedback_llm	bool	if true, a second LLM is used to analyze and summarize the evaluation log (e.g. compile errors)	
false

kernels_iter_0_path	string	
Useful for debugging, set to "best" to use the best kernel from the database for this task_name.

null
2a.4 Hierarchical Config Structure
The project uses Hydra (https://hydra.cc/) for hierarchical configuration management. This means that parameters are grouped in different topical sections. For example:

prompt:
  reference_language: SYCL

inference:
  servers:
   - _target_: kernelgen.inference_server.InferenceServer
     model_name: gpt-5.2
     temperature: 0.3
   - _target_: kernelgen.inference_server.InferenceServer
     model_name: gpt-5.3-codex
     temperature: 0.3

eval_config:
  warmup_min_time: 0.1
  profile_original_model: false

This config would (1) tell the prompt that the reference is given as SYCL code. (2) For LLM inference, this configuration would initialize one inference server with GPT 5.2 (temperature 0.3). (3) For evaluating the kernel, it would do 5 trials of the correctness check (comparing the kernel output to the reference output for randomized inputs) and would benchmark performance with a warmup time of 0.1 seconds. 

The config structure resembles a dictionary, but can also have lists (marked by dashes). For example, inference can be an ensemble of multiple language models, which might live on different servers, as shown in the example above.

The parameters of the individual parts will be explained in the following.

2a.5 Config parameters by section
LLM Inference ("inference" section)
Configures the LLM backend for code generation. Uses remote inference server (e.g., Intel GNAI). In the config, these parameters must be indented in the inference  section as shown above.

By default, inference uses an ensemble of multiple models, which can in principle be hosted on several servers. Thus. you need to provide the models as a list of servers. See example here.

Each inference server has the following parameters:

server_type	string	Type of inference server	intel_gnai
model_name	string	Model identifier	gpt-5.2
max_tokens	int	Maximum tokens to generate	5000
temperature	float	Sampling temperature (0.0 = deterministic)	0.0
num_completions	int	Number of completions to generate	1
verbose	bool	Enable verbose logging	False
timeout	int	Timeout for GNAI request (in seconds)	400
The most important inference parameter is the model_name. 

Important: The models available on GNAI change frequently. Check https://gpusw-docs.intel.com/services/gnai/models/ for an up-to-date list of available models.
Currently available on GNAI are: gpt-4.1,  gpt-4o, gpt-5-mini, gpt-5-nano, gpt-5.1, gpt-5.1-codex, gpt-5.1-codex-max, gpt-5.2, gpt-5.2-codex, o3, o3-mini, o4-mini, claude-4-5-sonnet, claude-4-5-sonnet-thinking, claude-4-5-haiku, claude-4-5-haiku-thinking, claude-4-5-opus, claude-4-5-opus-thinking. 

If you run into inference timeout issues, consider increasing the timeout.

Prompt ("prompt" section)
The prompt has several subsections. 

Top-level param.	Sub-level param.	Type	Description	Default
reference_language	
string	Programming language of the reference	Pytorch
num_optimization_tips	
int	Number of high-level optimization strategies to sample and include in the prompt	2
include_inspirations	
bool	Whether to include inspiration examples (prior generated kernels) in the prompt (only relevant it evolve_mode: true )	true
include_best_program	
bool	Whether to include the top program, i.e. the currently best kernel, in the prompt for reference (only relevant it evolve_mode: true )	true
include_hardware_specs	
bool	Whether to include hardware specs in the prompt	true
allow_templated	
bool	Whether to allow the model to write templated kernels with multiple parameter option (all implemented parameter options will be tested)	true
rag (list of objects)	


null
  - RagPytorchToSycl

top_k	int	Number of examples to use from the RAG database	1
(note: this is a database of thousands of SYCL kernels generated for KernelBench) 	restrict_to_level	int	If int is provided (1, 2, or 3), the database will only include examples from this KernelBench level	null
restrict_to_correct	bool	if true, only correct kernels can be included as an examples	true
min_runtime_improvement	float	Minimum speedup that a kernel must achieve to be included as an example (filtering for performant kernels)	1.1
rag_mode	string	Either "closest_match" (i.e. finding the example that is most related to the current task) or "random" (just including any performant kernel)	closest_match
include_prob	float	Probability to use RAG at all (if <1, only some of the prompts will have RAG examples)	1
 - RagESIMD	top_k	int	Number of examples to use from the RAG database	1
  (note: this is a database of ~80 ESIMD kernels plus a SYCL-ESIMD guide)

rag_mode	string	Either "closest_match" (i.e. finding the example that is most related to the current task) or "random" (just including any performant kernel) or "guide" (include only the guide, no examples)	closest_match
include_prob	float	Probability to use RAG at all (if <1, only some of the prompts will have RAG examples)	1
 - RagJointMatrix	top_k	int	Number of examples to use from the RAG database 	1
(note: this is a guide for using the SYCL joint matrix extension for XMX units. There are also 6 examples for joint matrix kernels)

rag_mode	string	Either "closest_match" (i.e. finding the example that is most related to the current task) or "random" (just including any performant kernel) or "guide" (include only the guide, no examples)	closest_match
include_prob	float	Probability to use RAG at all (if <1, only some of the prompts will have RAG examples)	1
  - RagHeCBench	top_k	int	Number of examples to use from the RAG database 	1
(note: this is a database of SYCL kernels from HeCBench)

rag_mode	string	Either "closest_match" (i.e. finding the example that is most related to the current task) or "random" (just including any performant kernel) or "guide" (include only the guide, no examples)	closest_match
include_prob	float	Probability to use RAG at all (if <1, only some of the prompts will have RAG examples)	1
max_length	int	Maximum string length of example	10000


Explanation:

The rag part of the prompt config consists of a list of RAG-database objects. The list has to be provided with specific paths to these classes in the hydra ListConfig format; e.g.

prompt:
  rag: 
    - _target_: kernelgen.database.rag_pytorch_to_sycl.RagPytorchToSycl
      top_k: 1 
      restrict_to_level: null 
      restrict_to_correct: True 
      min_runtime_improvement: 1.1 
      rag_mode: closest_match
    - _target_: kernelgen.database.rag_esimd.RagESIMD
      top_k: 1
      rag_mode: closest_match

and the same for kernelgen.database.rag_joint_matrix.RagJointMatrix and kernelgen.database.rag_hecbench.RagHeCBench with the parameters defined above.
Evolutionary algorithm ("database.config" section)
The use of the evolutionary algorithm is controlled via the evolve_mode  parameter on top level. If evolve_mode: true , the parameters of the evolutionary algorithm can be controlled in the database section. For example:

evolve_mode: true

database:
  config:
    exploration_ratio: 0.3
    num_top_programs: 2

num_top_programs	int	Number of top-performing programs to include	1
num_diverse_programs	int	Number of diverse programs to include	0
population_size	int	Total population size	1000
archive_size	int	Archive size for elites	100
num_islands	int	Number of evolutionary islands	4
programs_per_island	int	Programs per island before switching	10
num_inspirations	int	Number of inspiration examples	2
elite_selection_ratio	float	Ratio of elite selection	0.1
exploration_ratio	float	Exploration vs exploitation balance	0.2
exploitation_ratio	float	Exploitation ratio	0.7
diversity_metric	string	Metric for diversity (edit_distance, feature_based)	edit_distance
feature_dimensions	list	Dimensions for MAP-Elites	["complexity", "diversity"]
feature_bins	int/dict	Number of bins per dimension	10
diversity_reference_size	int	Reference set size for diversity	20
migration_interval	int	Migrate every N generations	10
migration_rate	float	Fraction of population to migrate	0.1



Created by Wiedemann, Nina, last updated by Melonakos, John on May 01, 2026  3 minute read
For running KernelFoundry for your kernel generation task, you need to define correctness tests and performance benchmarks. Our test framework is based on pytest. We provide fixtures and functions in the kernelfoundry package that simplify the development of kernel tests. 

Naming scheme: Tests must start with "test_" (as per pytest standards). Other than that, there are no naming conventions - all implemented tests will be assumed to be correctness tests, except if they are marked with the @pytest.mark.performance decorator (see 2b.2 below).

2b.1 Define correctness tests
At least one test is required. Tests must be contained in a class that inherits from CustomTest and must start with "test_".

A standard way that works for many use cases is to compare the kernel output to the reference output on random inputs, as in this example:


from kernelfoundry.testing import assert_allclose

def test_correctness(self, data, kernel):
    x, y = data
    result = kernel(x, y)
    expected_result = reference(x, y)
    assert_allclose(result, expected_result)
This test works as follows:

It uses a fixture to create random input data, e.g., two torch tensors x, y.
It then executes the kernel and the reference implementation on the input data.
The results are compared with the assert_allclose function provided in the kernelfoundry package.
You can use this test as-is for most custom tasks and only need to modify the reference function, the kernel fixture (importing the kernel module), and the data fixture.

2b.2 Define benchmarking tests
You must define one benchmarking function using the @pytest.mark.performance decorator. Currently, only one function is supported (any additional benchmarking functions will be executed but not used to compute the kernel's runtime and speedup).

The interface between your benchmarking test and the KernelFoundry framework is the profile_store fixture, which will store the results. We recommend using our standard benchmarking function measure_runtime, or, even simpler if you are using torch, measure_runtime_torch. This 

A standard way to benchmark your kernel is the following:

@pytest.mark.performance
def test_benchmark_my_kernel(self, kernel, device, data, measure_runtime_torch):
    # Run profiling on kernel
    measure_runtime_torch(kernel, device, args=data)
This test works as follows:

It uses a fixture to create random input data
It runs measure_runtime_torch, which handles 1) moving the input data to the GPU device, 2) warmup trials, and 3) storing results via the profile_store.
The alternative way is to write a custom benchmarking function that does not use measure_runtime. It must fulfill the following requirements to be compatible with the KernelFoundry code:

It must use the profile_store fixture to store the results in the form of a list of runtimes
It must use the profiler_session context to run the main trials - otherwise, profiling will not work because there is no entry point for the profiler to start
Examples for custom benchmarking are provided here within the matmul template.


2b.3 Debugging the tests
If you have the kernelfoundry package installed (see 1. Getting Started), you can debug the tests locally on your machine.

To test whether your tests pass with the reference, run

pytest --ref -s task.py
This will execute all tests with the reference instead of the kernel. 



To test whether the build works, you need to already have a correct kernel in the EVOLVE section. Navigate to the folder and run 

python task.py
This will execute the build function of your CustomTest class and compile the kernel to a shared object. If the kernel compiled, run the following to test the kernel.

pytest -s task.py


Created by Wiedemann, Nina, last updated by Melonakos, John on Apr 27, 2026  2 minute read
A simple UI-based way to verify whether your task definition is correct is to use the "Validate KernelFoundry task" button in VSCode extension or Web UI (see 3a. Running a Task via the VSCode Extension). However, waiting for the result and retrying can be cumbersome when there are difficulties in defining the tests. Instead, you can install the kernelfoundry package to debug locally.

KernelFoundry is a lightweight Python package for compilation and testing. For installation instructions, see 1. Getting Started. Documentation can be found here.

Compile a custom task locally:
In the provided templates, we added a __main__ block to compile the reference and kernel (see here for example). Navigating to the directory of the custom task and run:



python task.py
If your build function is defined correctly, a shared object will be created (or two if the reference also needs to be built) and stored in the same folder.

NOTE: If you only provide a skeleton / placeholder code for the initial kernel (i.e. the EVOLVE block), the kernel build may fail! This is expected and does not mean that the task is invalid. 

Testing the reference:
The most important test for the validity of a task is that the reference implementation passes the tests. To run the tests on the reference, navigate to the directory of the custom task and execute:



pytest --ref task.py
Testing the custom kernel:
If you provide a functional kernel in the EVOLVE-block, you can test this kernel with:



pytest task.py


Created by Wiedemann, Nina, last updated on Apr 30, 2026  2 minute read
3.1. Workflow
The standard workflow is as follows:

Define a custom task by adapting a template or writing tests and references from scratch (see 2. Defining a Custom Task)
Validate the task via the VSCode extension OR the KernelFoundry web app → Check the Logs whether it works as desired
Run the kernel generation pipeline via the VSCode extension OR the KernelFoundry web app → Check the Results to see your generated kernels
3.2. What are tasks and jobs?
A task is a specific kernel generation project, defined by one operation for which you would like to write a kernel. Once you have defined the task, you can start different jobs. For example, one job could be a test with a small number of iterations and a cheap language model, while the next job could use more iterations and the latest, most powerful language models.

Task
task_name (e.g. gemm_int8)
Evolve-block
Reference implementation
Job 1
job_name = test
Parameters:
Number of iterations = 1
Number of branches = 2
Model name = GPT 4.1 
Job 2
job_name = performance
Parameters:
Number of iterations = 5
Number of branches = 4
Model name = Claude-Sonnest
...
Iteration 1
Iteration 2
...
k1
Evolutionary
kernel
generation
k2
k3
k4
k1
k2
k5
k6
k3
k4
k7
k8
Kernel 8
Kernel code
Evaluation log
Profiling result
A simplified overview of the KernelFoundry workflow is shown below. In short, the prompt is initially constructed from the user input (reference implementation, user instructions) and inspirations from the RAG knowledge base. In further iterations, a kernel is sampled from a database that contains all previously generated kernels ("parent"), and the prompt asks the LLM to improve this kernel based on the evaluation log including profiling output.

Prompt













User instructions
Reference
RAG / Examples / Inspirations
Initial code / skeleton
task.py

[USER_INSTRUCTIONS_START]
Some additional instructions
[USER_INSTRUCTIONS_START]

[REFERENCE_START]
Reference code
[REFERENCE_START]

Pytest correctness & benchmark
kernel.sycl

<Model Code>
// [EVOLE START]
Kernel code
// [EVOLVE END]
Further model code
Iteration 1
LLM
Testing & Benchmarking
Prompt













User instructions
Reference
RAG / Examples / Inspirations
Previous kernel + Evaluation log
Best kernel generated so far
Database of generated kernels
RAG database
Iteration 2 and beyond
sample. parent
Input
LLM
Testing & Benchmarking
3.3. How to get the best performance?
Once you have defined a custom task and validated it (ensuring the reference is correct and the tests work as intended), it is time to submit several jobs. To achieve the best performance, adjust the configuration parameters. The starting point is the config files provided in the templates, which are kept simple and just run the most basic version of KernelFoundry, improving the kernel over three iterations. To advance from there, use the following features of KernelFoundry:

1) Start from best kernel: If you start a new job but have already generated a good kernel in a previous job, you can always start from the best kernel generated so far (for this task_name and gpu_arch) by adding the following to the config:



kernels_iter_0_path: best
2) Evolutionary algorithm with more iterations & more branches: Activate evolutionary mode (random sampling of the parent kernel, multiple branches) and increase the number of iterations, by adding to the config:



evolve_mode: true
branches_per_iteration: 3
max_iters: 15
3) Use better & more diverse language models: The best strategy is to use an ensemble of several language models, e.g., claude-4-5-sonnet, gpt-5.3, and gpt-5.1-codex-max (see here for an example of how to specify in the config).

4) Inject knowledge with RAG (experimental): You can try to inject knowledge with RAG. We provide a database of high-performance SYCL kernels, a guide to joint matrices, and examples of ESIMD kernels.




prompt:
  rag:
    - _target_: kernelgen.database.rag_pytorch_to_sycl.RagPytorchToSycl
      top_k: 1
      min_runtime_improvement: 1.1 # only include SYCL kernels in the prompt that have >1.1 speedup
	- _target_: kernelgen.database._target_: kernelgen.database.rag_joint_matrix.RagJointMatrix
      include_prob: 1
5) Fix correct kernels: If you observe many incorrect kernels, it could help to use a FeedbackLLM that interprets the evaluation log: 



use_feedback_llm: true
Check out 2a. Config Parameters for further options, and use our config generator to generate the config file based on your choices.