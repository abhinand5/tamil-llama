import argparse
import os
import sys

import bitsandbytes as bnb
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

"""
python make_shards.py \
    --model_name your_model_name \
    --save_model_name hf_user/hf_model_name \
    --max_shard_size 2GB \
    --push_to_hub \
    --push_mode_private
"""


def main():
    parser = argparse.ArgumentParser(
        description="Push a sharded model and tokenizer to the Hub."
    )

    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the model to load and push to the Hub",
    )
    parser.add_argument(
        "--save_model_name", default=None, help="Name of the sharded model"
    )
    parser.add_argument("--token", default=None, help="Huggingface Hub Token")
    parser.add_argument(
        "--max_shard_size", default="1GB", help="Maximum shard size for pushing"
    )
    parser.add_argument(
        "--push_to_hub", default=True, action="store_true", help="Push the model to hub"
    )
    parser.add_argument(
        "--use_fp16", default=True, action="store_true", help="Use fp16"
    )
    parser.add_argument(
        "--push_mode_private",
        default=True,
        action="store_true",
        help="Push mode (private or public)",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.token is not None:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.token
    else:
        print("No HF token provided")
        sys.exit(1)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not args.use_fp16 else torch.float16,
        device_map="auto",
        offload_folder="offload",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.save_model_name is None:
        save_model = (
            f"{args.model_name.split('/')[1]}-sharded-{args.max_shard_size}"
        )
    else:
        save_model = args.save_model_name

    if args.push_to_hub:
        model.push_to_hub(
            save_model,
            max_shard_size=args.max_shard_size,
            private=args.push_mode_private,
            token=args.token,
        )
        tokenizer.push_to_hub(
            save_model, private=args.push_mode_private, token=args.token
        )


if __name__ == "__main__":
    main()
