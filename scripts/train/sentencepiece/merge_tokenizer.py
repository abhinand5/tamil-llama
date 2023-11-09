# Based on https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizer/merge_tokenizers.py
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import argparse

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import LlamaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--llama_tokenizer_dir", default=None, type=str, required=True)
parser.add_argument("--tamil_sp_model_file", default="./tamil_sp.model", type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
tamil_sp_model_file = args.tamil_sp_model_file

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
tamil_sp_model = spm.SentencePieceProcessor()
tamil_sp_model.Load(tamil_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
tamil_spm = sp_pb2_model.ModelProto()
tamil_spm.ParseFromString(tamil_sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer), len(tamil_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

## Add Tamil tokens to LLaMA tokenizer
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
for p in tamil_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir = "merged_tokenizer_sp"
output_hf_dir = "merged_tokenizer_hf"  # the path to save Tamil-LLaMA tokenizer
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir + "/tamil_llama.model", "wb") as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + "/tamil_llama.model")

tokenizer.save_pretrained(output_hf_dir)
print(f"Tamil-LLaMA tokenizer has been saved to {output_hf_dir}")


# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
tamil_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text = """கோப்பையை உறுதி செய்யுமா இந்தியா?
Can India secure the World Cup trophy?"""
print("Test text:\n", text)
llama_tokenized = llama_tokenizer.tokenize(text)
tamil_llama_tokenized = tamil_llama_tokenizer.tokenize(text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenized}")
print(f"LLaMA tokenizer n_tokens={len(llama_tokenized)}")
print(f"Tokenized by Tamil-LLaMA tokenizer:{tamil_llama_tokenized}")
print(f"Tamil LLaMA tokenizer n_tokens={len(tamil_llama_tokenized)}")
