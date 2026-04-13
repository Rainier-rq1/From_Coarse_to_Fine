set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ray stop
# ray start --head --num-gpus 8

# MODEL_PATH=/nas/shared/kilab/hf-hub/cold_ckpt
#MODEL_PATH=/oss/rqy/checkpoints/if_all_dp_nc/qwen2_5_7b/actor/huggingface

#MODEL_PATH=/nas/shared/kilab/hf-hub/DeepScaleR-1.5B-Preview
MODEL_PATH=/mnt/shared-storage-user/evobox-share/hf-hub/models--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=8 \
    



    
