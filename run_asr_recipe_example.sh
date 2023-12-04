#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=15         # CPU cores/threads
#SBATCH --mem-per-cpu=1gb
#SBATCH --output=logtest.log
#SBATCH --gpus-per-node=1

export CUDA_VISIBLE_DEVICES="0"
export CUDA_LAUNCH_BLOCKING=1

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.asr

# Create Deepspeed configuration file
config_json=ds_config.json
cat <<EOT > $config_json  
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

   "scheduler": {
         "type": "WarmupDecayLR",
         "params": {
             "last_batch_iteration": -1,
             "total_num_steps": "auto",
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
EOT
deepspeed --num_nodes 1 --num_gpus 1 run_asr_recipe.py --deepspeed --deepspeed_config $config_json 
