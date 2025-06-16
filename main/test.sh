#! /bin/bash

# Test scripts
python main/test_time_scaling.py run_name=test_base method="base" prompt="regular" dataset_src=local level=1 subset="(1, 4)"
# python main/test_time_scaling.py run_name=test_cot method="base" prompt="cot" dataset_src=local level=1 subset="(1, 4)" server_type="openai" model_name="gpt-4o-mini" 
# python main/test_time_scaling.py run_name=test_best_of_n method="best-of-N" num_parallel=4 dataset_src=local level=1 subset="(1, 4)" verbose=True log_prompt=True num_workers=4 server_type="openai" model_name="gpt-4o-mini" num_cpu_workers=4 num_gpu_devices=2
# python main/test_time_scaling.py run_name=test_iterative_refinement method="iterative refinement" num_iterations=4 dataset_src=local level=1 subset="(1, 4)" verbose=True log_prompt=True num_workers=4 server_type="openai" model_name="gpt-4o-mini" num_cpu_workers=4 num_gpu_devices=2
# python main/test_time_scaling.py run_name=test_metr method="METR" num_parallel=1 num_samples=4 dataset_src=local level=1 subset="(1, 4)" verbose=True log_prompt=True num_workers=4 server_type="openai" model_name="gpt-4o-mini" num_cpu_workers=4 num_gpu_devices=2
# python main/test_time_scaling.py run_name=test_stanford method="Stanford" num_parallel=4 num_iterations=4 num_best=1 dataset_src=local level=1 subset="(1, 4)" verbose=True log_prompt=True num_workers=4 server_type="openai" model_name="gpt-4o-mini" num_cpu_workers=4 num_gpu_devices=2