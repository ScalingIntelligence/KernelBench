# # Metrics
# python main/metrics.py --run_dir runs/base_level1 --hardware A6000_babel
# python main/metrics.py --run_dir runs/base_level2 --hardware A6000_babel
# python main/metrics.py --run_dir runs/base_level3 --hardware A6000_babel
# python main/metrics.py --run_dir runs/best_of_n_level1 --hardware A6000_babel
# python main/metrics.py --run_dir runs/best_of_n_level2 --hardware A6000_babel
# python main/metrics.py --run_dir runs/best_of_n_level3 --hardware A6000_babel
# python main/metrics.py --run_dir runs/IR_level1 --hardware A6000_babel
# python main/metrics.py --run_dir runs/IR_level2 --hardware A6000_babel
# # python main/metrics.py --run_dir runs/IR_level3 --hardware A6000_babel
# python main/metrics.py --run_dir runs/metr_level1 --hardware A6000_babel
# python main/metrics.py --run_dir runs/metr_level2 --hardware A6000_babel
# # python main/metrics.py --run_dir runs/metr_level3 --hardware A6000_babel


# Analysis
python main/analysis.py --method base
python main/analysis.py --method best_of_n
python main/analysis.py --method IR
python main/analysis.py --method metr

python main/analysis.py --level 1 --methods "base,best_of_n,IR,metr"
python main/analysis.py --level 2 --methods "base,best_of_n,IR,metr"
python main/analysis.py --level 3 --methods "base,best_of_n,IR,metr"
