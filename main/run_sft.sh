set -x

RUN_NAME="sft_Qwen2.5-7B-Instruct_best_1_with_reasoning"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/KernelBench/KernelBench/sft_dataset_best_1_train_with_reasoning.parquet \
    data.val_files=$HOME/KernelBench/KernelBench/sft_dataset_best_1_eval_with_reasoning.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.train_batch_size=4 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=16384 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    model.target_modules=all-linear \
    model.lora_rank=64 \
    model.lora_alpha=16 \
    trainer.project_name=KernelBench \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=/data/user_data/gyeongwk/KernelBench/sft/$RUN_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb'] 