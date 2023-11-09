import argparse
import logging
import os

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CorpusCreator:
    def __init__(self):
        self.output_dir = "./corpus"

    def create_sentence_corpus(
        self,
        hf_dataset,
        text_col,
        dataset_split="train",
        output_file_name="tamil_sentence_corpus.txt",
    ):
        try:
            dataset = load_dataset(hf_dataset, split=dataset_split)
            train_df = pd.DataFrame(dataset)

            os.makedirs(self.output_dir, exist_ok=True)
            corpus_path = os.path.join(self.output_dir, output_file_name)

            with open(corpus_path, "w") as file:
                for index, value in tqdm(
                    train_df[text_col].iteritems(), total=len(train_df)
                ):
                    file.write(str(value) + "\n")

        except Exception as e:
            logger.error(f"Error creating the text corpus -> {e}")

        return corpus_path

    def run(self):
        parser = argparse.ArgumentParser(
            description="Create a sentence corpus from a Hugging Face dataset."
        )
        parser.add_argument(
            "--hf-dataset",
            required=True,
            help="Name of the Hugging Face dataset (e.g., 'imdb').",
        )
        parser.add_argument(
            "--text-col",
            required=True,
            help="Name of the text column in the dataset.",
        )
        parser.add_argument(
            "--dataset-split",
            default="train",
            help="Dataset split to use (default: 'train').",
        )
        parser.add_argument(
            "--output-file-name",
            default="tamil_sentence_corpus.txt",
            help="Name of the output corpus file (default: 'tamil_sentence_corpus.txt').",
        )

        args = parser.parse_args()
        self.create_sentence_corpus(
            args.hf_dataset, args.text_col, args.dataset_split, args.output_file_name
        )


if __name__ == "__main__":
    corpus_creator = CorpusCreator()
    corpus_creator.run()
