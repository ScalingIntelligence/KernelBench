# Metrics
python main/metrics.py --hardware A6000_babel --run_dir runs/base_level1_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/base_level2_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/base_level3_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/base_level5_DeepSeek-R1

python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level1_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level2_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level3_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level5_DeepSeek-R1

python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level1_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level2_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level3_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level5_DeepSeek-R1

python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level1_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level2_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level3_DeepSeek-R1
python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level5_DeepSeek-R1

python main/metrics.py --hardware A6000_babel --run_dir runs/base_level1_Qwen2.5-7B-Instruct
python main/metrics.py --hardware A6000_babel --run_dir runs/base_level2_Qwen2.5-7B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/base_level3_Qwen2.5-7B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/base_level5_Qwen2.5-7B-Instruct

python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level1_Qwen2.5-7B-Instruct
python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level2_Qwen2.5-7B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level3_Qwen2.5-7B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level5_Qwen2.5-7B-Instruct

python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level1_Qwen2.5-7B-Instruct
python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level2_Qwen2.5-7B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level3_Qwen2.5-7B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level5_Qwen2.5-7B-Instruct

python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level1_Qwen2.5-7B-Instruct
python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level2_Qwen2.5-7B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level3_Qwen2.5-7B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level5_Qwen2.5-7B-Instruct

python main/metrics.py --hardware A6000_babel --run_dir runs/base_level1_Qwen2.5-1.5B-Instruct
python main/metrics.py --hardware A6000_babel --run_dir runs/base_level2_Qwen2.5-1.5B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/base_level3_Qwen2.5-1.5B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/base_level5_Qwen2.5-1.5B-Instruct

python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level1_Qwen2.5-1.5B-Instruct
python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level2_Qwen2.5-1.5B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level3_Qwen2.5-1.5B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/best_of_n_level5_Qwen2.5-1.5B-Instruct

python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level1_Qwen2.5-1.5B-Instruct
python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level2_Qwen2.5-1.5B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level3_Qwen2.5-1.5B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/IR_level5_Qwen2.5-1.5B-Instruct

python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level1_Qwen2.5-1.5B-Instruct
python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level2_Qwen2.5-1.5B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level3_Qwen2.5-1.5B-Instruct
# python main/metrics.py --hardware A6000_babel --run_dir runs/metr_level5_Qwen2.5-1.5B-Instruct

# GRPO
# python main/metrics.py --hardware A6000_babel --run_dir runs/grpo_train_Qwen2.5-7B-Instruct --grpo

python main/metrics.py --hardware A6000_babel --run_dir runs/grpo_eval_level1_Qwen2.5-7B-Instruct
python main/metrics.py --hardware A6000_babel --run_dir runs/grpo_eval_level2_Qwen2.5-7B-Instruct



# Analysis
# python main/analysis.py --axis method --level 1 --model deepseek_r1
# python main/analysis.py --axis method --level 2 --model deepseek_r1
# python main/analysis.py --axis method --level 3 --model deepseek_r1
# python main/analysis.py --axis method --level 5 --model deepseek_r1
# python main/analysis.py --axis method --level 1 --model qwen_2.5_7b
# python main/analysis.py --axis method --level 2 --model qwen_2.5_7b
# # python main/analysis.py --axis method --level 3 --model qwen_2.5_7b
# # python main/analysis.py --axis method --level 5 --model qwen_2.5_7b
# python main/analysis.py --axis method --level 1 --model qwen_2.5_1.5b
# python main/analysis.py --axis method --level 2 --model qwen_2.5_1.5b
# # python main/analysis.py --axis method --level 3 --model qwen_2.5_1.5b
# # python main/analysis.py --axis method --level 5 --model qwen_2.5_1.5b

# python main/analysis.py --axis level --method base --model deepseek_r1
# python main/analysis.py --axis level --method best_of_n --model deepseek_r1
# python main/analysis.py --axis level --method IR --model deepseek_r1
# python main/analysis.py --axis level --method metr --model deepseek_r1
# python main/analysis.py --axis level --method base --model qwen_2.5_7b
# python main/analysis.py --axis level --method best_of_n --model qwen_2.5_7b
# python main/analysis.py --axis level --method IR --model qwen_2.5_7b
# python main/analysis.py --axis level --method metr --model qwen_2.5_7b
# python main/analysis.py --axis level --method base --model qwen_2.5_1.5b
# python main/analysis.py --axis level --method best_of_n --model qwen_2.5_1.5b
# python main/analysis.py --axis level --method IR --model qwen_2.5_1.5b
# python main/analysis.py --axis level --method metr --model qwen_2.5_1.5b


# python main/analysis.py --axis model --method base --level 1
# python main/analysis.py --axis model --method base --level 2
# python main/analysis.py --axis model --method base --level 3
# python main/analysis.py --axis model --method base --level 5
# python main/analysis.py --axis model --method best_of_n --level 1
# python main/analysis.py --axis model --method best_of_n --level 2
# python main/analysis.py --axis model --method best_of_n --level 3
# python main/analysis.py --axis model --method best_of_n --level 5
# python main/analysis.py --axis model --method IR --level 1
# python main/analysis.py --axis model --method IR --level 2
# python main/analysis.py --axis model --method IR --level 3
# python main/analysis.py --axis model --method IR --level 5
# python main/analysis.py --axis model --method metr --level 1
# python main/analysis.py --axis model --method metr --level 2
# python main/analysis.py --axis model --method metr --level 3
# python main/analysis.py --axis model --method metr --level 5