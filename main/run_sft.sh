set -x

RUN_NAME="sft_Qwen2.5-7B-Instruct_best_3"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/KernelBench/KernelBench/sft_dataset_best_3_train.parquet \
    data.val_files=$HOME/KernelBench/KernelBench/sft_dataset_best_3_eval.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=4096 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    model.target_modules=all-linear \
    model.lora_rank=64 \
    model.lora_alpha=16 \
    trainer.project_name=KernelBench \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=/data/user_data/gyeongwk/KernelBench/sft/$RUN_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=50 \
    trainer.save_freq=660 \
    trainer.logger=['console','wandb'] 