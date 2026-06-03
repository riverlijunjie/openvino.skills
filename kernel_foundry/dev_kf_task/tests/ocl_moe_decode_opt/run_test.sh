
kernel_foundry_path=/mnt/river/kernel_foundry/kernelfoundry.internal
test_task_path=/mnt/river/kernel_foundry/workspace/ocl_moe_decode_opt

cd $test_task_path
pytest --ref -s $test_task_path/task.py
pytest -s $test_task_path/task.py
cd -