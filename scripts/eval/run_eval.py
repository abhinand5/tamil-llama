import argparse
import ast
import json
import time

import datasets
import pandas as pd
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from openai.error import RateLimitError
from pydantic import BaseModel, validator

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

class ScoringTemplate(BaseModel):
    score: float
    reason: str

    @validator("score")
    def score_in_range(cls, v):
        if not (0 <= v <= 10):
            raise ValueError("Incorrect Score!")
        return v


template = "You will be given a ChatGPT-like systems' outputs. Please rate an overall score on a ten-point scale for each and give explanations to justify your scores. Give your output in form of a python dictionary, the response should ONLY contain a python dictionary, we'll use your code directly in our code so make sure the dict has one field called score and another field called reason inside which you put in a brief explanation of your score."
human_template = "{model_output}"

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", human_template),
    ]
)


def predict(model, prompt, idx="UNK"):
    for i in range(MAX_RETRIES):
        try:
            return model.predict_messages(prompt).content.rstrip()
        except RateLimitError as e:
            print(
                f"[{idx}] Rate Limit Error occurred! Retrying - {i+1}/{MAX_RETRIES} after {MIN_WAIT_TIME}s..."
            )
            time.sleep(MIN_WAIT_TIME)
            continue
        except Exception as e:
            print(f"[{idx}] Error while predicting -> {e}")
            return None
    return None


def generate(sample, idx):
    model_output = sample[MODEL_OUTPUT_FIELD]
    prompt = chat_prompt.format_messages(model_output=model_output)
    model = ChatOpenAI(temperature=0.2, model_name="gpt-4", max_retries=0, cache=True)
    resp = predict(model, prompt, idx=idx)

    if resp is not None:
        try:
            parsed_resp = ast.literal_eval(resp)
            return resp
        except Exception as e:
            print(f"[{idx}] Error parsing output dict from model -> {e}")
            return resp
    return resp


def add_scores(sample, idx):
    try:
        output = generate(sample, idx)
        output_dict = json.loads(output)
        sample[f"{MODEL_OUTPUT_FIELD}_score"] = output_dict["score"]
        sample[f"{MODEL_OUTPUT_FIELD}_reason"] = output_dict["reason"]
    except Exception as e:
        print(f"Error saving outputs -> {e}")
        sample[f"{MODEL_OUTPUT_FIELD}_score"] = -1
        sample[f"{MODEL_OUTPUT_FIELD}_reason"] = ""

    return sample


def main(input_csv, output_csv):
    global MODEL_OUTPUT_FIELD, MAX_RETRIES, MIN_WAIT_TIME
    MODEL_OUTPUT_FIELD = model_output_field
    MAX_RETRIES = max_retries
    MIN_WAIT_TIME = min_wait_time

    eval_set = datasets.load_dataset("csv", data_files=input_csv, split="train")
    eval_set = eval_set.map(add_scores, with_indices=True, num_proc=2)
    df = pd.DataFrame(eval_set)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and score model outputs.")
    parser.add_argument(
        "--input-csv",
        help="Path to the input CSV file with model outputs and instructions.",
        required=True,
    )
    parser.add_argument(
        "--output-csv",
        help="Path to save the output CSV file with scores.",
        default="tamil_eval_scores.csv",
    )
    parser.add_argument(
        "--model-output-field",
        help="Field name of the model output in the CSV file.",
        default="tamil-llama",
    )
    parser.add_argument(
        "--max-retries",
        help="Maximum number of retries on RateLimitError.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min-wait-time",
        help="Minimum wait time (in seconds) before retrying after RateLimitError.",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    main(input_csv=args.input_csv, output_csv=args.output_csv)
