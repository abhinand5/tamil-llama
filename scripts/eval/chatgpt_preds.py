import argparse
import json
from multiprocessing.pool import ThreadPool
import pandas as pd
import datasets
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from tqdm import tqdm
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from enum import Enum

# Set up the cache for langchain
set_llm_cache(SQLiteCache(database_path=".langchain.db"))


# Enum class for different colors for printing
class Colors(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GREY = "\033[90m"
    RESET = "\033[0m"


# Function to print text with color
def print_color(text, color, *args, **kwargs):
    print(f"{color.value}{text}{Colors.RESET.value}", *args, **kwargs)


# Define prompt templates
PROMPT_TEMPLATES = {
    "ta": ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are an assistant fluent in Tamil. Respond clearly, truthfully, and concisely to user instructions in Tamil."
            ),
            HumanMessagePromptTemplate.from_template("{instruction}"),
        ]
    ),
}


# Function to generate the response
def generate(sample, verbose):
    instruction = sample["Task"]
    prompt = PROMPT_TEMPLATES[cur_prompt].format_messages(instruction=instruction)
    chat = ChatOpenAI(
        temperature=0.2, model_name="gpt-3.5-turbo", max_retries=2, cache=True
    )
    resp = chat.predict_messages(prompt).content.rstrip()

    if verbose:
        print_color(f"( Human: ", Colors.GREEN, end="")
        print(instruction)
        print_color(f"(GPT: ", Colors.RED, end="")
        print_color("-------------------------", Colors.GREY)
        print(resp)
        print_color("--------------------------------", Colors.GREY)

    return resp


# Function to add predictions to the sample
def add_preds(sample, verbose):
    resp = generate(sample, verbose)
    sample["gpt3.5_turbo"] = resp
    return sample


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ChatGPT predictions for a given dataset"
    )
    parser.add_argument(
        "--instructions_csv_path",
        type=str,
        default="./preds/tamil_alpaca_eval.csv",
        help="Path to the CSV file containing the instructions",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./preds/tamil_eval_preds_gpt.csv",
        help="Path to save the predictions CSV file",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads to use for processing",
    )

    args = parser.parse_args()

    cur_prompt = "ta"
    VERBOSE = args.verbose
    N_THREADS = args.num_threads
    SAVE_PATH = args.save_path

    eval_set = datasets.load_dataset(
        "csv", data_files=args.instructions_csv_path, split="train"
    )
    eval_set = eval_set.map(
        lambda sample: add_preds(sample, VERBOSE), num_proc=N_THREADS
    )
    df = pd.DataFrame(eval_set)
    df.to_csv(SAVE_PATH, index=False)
