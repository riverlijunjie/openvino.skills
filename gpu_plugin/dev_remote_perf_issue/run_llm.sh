source ~/openvino/install_release/setupvars.sh

model_dir=~/river/models/qwen3-moe-4x0.6b-2.4b_1105/pytorch/ov/OV_FP16-4BIT_DEFAULT
prompts_dir=~/river/frameworks.ai.openvino.llm.prompts/2048/qwen3-30b-a3b.jsonl

python3 benchmark.py -d GPU -pf $prompts_dir -m $model_dir -mc 1 -ic 32 -n 1