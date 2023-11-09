import argparse
import os

import sentencepiece as spm


class SentencePieceTrainer:
    def __init__(self):
        self.corpus_dir = "./corpus"
        self.output_dir = "./models"
        self.model_prefix = "tamil_sp"
        self.vocab_size = 20000
        self.character_coverage = 1.0
        self.model_type = "unigram"

    def train_sentencepiece_model(self, input_file):
        output_model_path = os.path.join(self.output_dir, f"{self.model_prefix}.model")

        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            model_type=self.model_type,
        )

        os.rename(
            f"{self.model_prefix}.vocab",
            os.path.join(self.output_dir, f"{self.model_prefix}.vocab"),
        )
        os.rename(
            f"{self.model_prefix}.model",
            os.path.join(self.output_dir, f"{self.model_prefix}.model"),
        )

        return output_model_path

    def run(self):
        parser = argparse.ArgumentParser(description="Train a SentencePiece model.")
        parser.add_argument(
            "--input-file",
            required=True,
            help="Path to the input text corpus file.",
        )
        parser.add_argument(
            "--output-dir",
            default=self.output_dir,
            help="Directory where the trained model and vocabulary will be saved.",
        )
        parser.add_argument(
            "--model-prefix",
            default=self.model_prefix,
            help="Prefix for the model and vocabulary filenames.",
        )
        parser.add_argument(
            "--vocab-size",
            type=int,
            default=self.vocab_size,
            help="Size of the vocabulary.",
        )
        parser.add_argument(
            "--character-coverage",
            type=float,
            default=self.character_coverage,
            help="Character coverage for the model.",
        )
        parser.add_argument(
            "--model-type",
            default=self.model_type,
            choices=["bpe", "unigram", "char", "word"],
            help="Type of SentencePiece model.",
        )

        args = parser.parse_args()
        self.output_dir = args.output_dir
        self.model_prefix = args.model_prefix
        self.vocab_size = args.vocab_size
        self.character_coverage = args.character_coverage
        self.model_type = args.model_type

        os.makedirs(self.output_dir, exist_ok=True)

        self.train_sentencepiece_model(args.input_file)


if __name__ == "__main__":
    sp_trainer = SentencePieceTrainer()
    sp_trainer.run()
