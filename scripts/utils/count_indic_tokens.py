import argparse
from transformers import AutoTokenizer

# This is NOT perfect but gives high level idea about the composition of the tokenizer
language_unicode_ranges = {
    'European': ('\u0000', '\u007F'),
    'Chinese (Basic)': ('\u4E00', '\u9FFF'),
    'Tamil': ('\u0B80', '\u0BFF'),
    'Hindi': ('\u0900', '\u097F'),
    'Telugu': ('\u0C00', '\u0C7F'),
    'Malayalam': ('\u0D00', '\u0D7F'),
    'Kannada': ('\u0C80', '\u0CFF'),
    'Marathi': ('\u0900', '\u097F'),  # Marathi shares the range with Hindi
    'Bengali': ('\u0980', '\u09FF'),
}

def is_language(token, ranges):
    return any(ranges[0] <= char <= ranges[1] for char in token)

def count_language_tokens(tokenizer, ranges):
    return sum(is_language(token, ranges) for token in tokenizer.get_vocab().keys())

def main(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    total_vocab_size = len(tokenizer.get_vocab())

    print("\n---Note: These calculations are approximate!---\n")
    print(f"Total vocabulary size of '{model_name}': {total_vocab_size}\n")
    print(f"{'Language':<20} | {'Tokens':>10} | {'Percentage':>10}")
    print("-" * 50)

    for language, ranges in language_unicode_ranges.items():
        count = count_language_tokens(tokenizer, ranges)
        percentage = (count / total_vocab_size) * 100
        print(f"{language:<20} | {count:>10} | {percentage:>9.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count language-specific tokens and their percentage in a tokenizer's vocabulary.")
    parser.add_argument("model_name", type=str, help="Name of the model to analyze")

    args = parser.parse_args()

    main(args.model_name)