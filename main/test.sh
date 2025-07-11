#! /bin/bash

# Test scripts
python main/test_time_scaling.py --run_name test_base --method="base" --level=1 --subset="(1,1)" --server_type=openai --model_name="gpt-4o-mini" 
python main/test_time_scaling.py run_name=test_cot method="base" prompt="cot" dataset_src=local level=1 subset="(1, 4)" 
python main/test_time_scaling.py run_name=test_best_of_n method="best-of-N" num_parallel=2 dataset_src=local level=1 subset="(1, 4)" num_workers=4 num_cpu_workers=4 num_eval_devices=2
python main/test_time_scaling.py run_name=test_iterative_refinement method="iterative refinement" num_iterations=2 dataset_src=local level=1 subset="(1, 4)" num_workers=4 num_cpu_workers=4 num_eval_devices=2
python main/test_time_scaling.py run_name=test_metr method="METR" num_parallel=1 num_samples=2 dataset_src=local level=1 subset="(1, 4)" num_workers=4 num_cpu_workers=4 num_eval_devices=2