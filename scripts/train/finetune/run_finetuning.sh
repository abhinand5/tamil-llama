#!/bin/bash
echo "=== START TRAINING SCRIPT ==="
python finetune.py
echo "=== END TRAINING SCRIPT ==="
sleep 30
echo "=== START MAKING SHARDS ==="
python make_shards.py \
    --model_name ./abhinand/tamil-llama-7b-instruct-v0.1 \
    --save_model_name abhinand/tamil-llama-7b-instruct-v0.1-sharded \
    --max_shard_size 2GB \
    --push_to_hub \
    --push_mode_private
echo "=== END MAKING SHARDS ==="
# Might be helpful if you're using Azure
# echo "=== SHUTTING DOWN INSTANCE IN 30s ==="
# az vm deallocate -g "resource-group-name" -n "vm-name"
