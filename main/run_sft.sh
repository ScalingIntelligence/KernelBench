set -x

torchrun -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/KernelBench/KernelBench/sft_dataset_best_1_train.parquet \
    data.val_files=$HOME/KernelBench/KernelBench/sft_dataset_best_1_eval.parquet \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    trainer.default_local_dir=/data/user_data/gyeongwk/KernelBench/checkpoints/sft \
    trainer.project_name=KernelBench \
    trainer.experiment_name=SFT-Qwen2.5-7B-Instruct \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb']