# Tamil-Llama: A Family of LLaMA-based LLMs focused on the Tamil Language

<img src="assets/introducing_tamil_llama.png" alt="Tamil LLaMA Image" width="300" height="auto">

## Description

This repository contains the code and models for "Tamil-Llama", a project focused on enhancing the performance of language models for the Tamil language. It builds upon the open-source LLaMA model, introducing additional Tamil tokens and employing the LoRA methodology for efficient training. Please take a look at the technical report for more details.

Technical Report: [https://arxiv.org/abs/2311.05845](https://arxiv.org/abs/2311.05845)

If you appreciate this work and want to support its continued development, consider [buying me a coffee](https://www.buymeacoffee.com/abhinand.b). Your support is invaluable and greatly appreciated.

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/abhinand.b)

## Updates

### Feb 25, 2024

Google's Gemma 2B Model was adapter for Tamil (Experimental Release) based on the same framework with a few changes. More info in [this](https://www.linkedin.com/posts/abhinand-05_%3F%3F%3F%3F%3F%3F%3F%3F%3F%3F%3F-%3F%3F%3F%3F%3F-%3F%3F-activity-7167767094619430912-VspR?utm_source=share&utm_medium=member_desktop) LinkedIn post.

> **Note:** I have migrated to [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) for pretraining and [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for finetuning.

- No expansion in vocab for Gemma as it already has 256k vocab size and minnescule amounts of Tamil tokens.
- Continually pretrain on all available Tamil Wikipedia data for 3 epochs.
- Finetune on Tamil Alpaca + English Alpaca mix for 5 epochs
- Model tops Open LLM Leaderboard for models under 3B params as of Feb 2023.

**Download Links:**

- [Tamil Gemma 2B Alpha](https://huggingface.co/abhinand/gemma-2b-it-tamil-v0.1-alpha) 
- [Tamil Gemma 2B Alpha GGUF](https://huggingface.co/abhinand/gemma-2b-it-tamil-v0.1-alpha-GGUF)

### Jan 23, 2024 

For more details, please read the detailed blog post [here](https://abhinand05.medium.com/breaking-language-barriers-introducing-tamil-llama-v0-2-and-its-expansion-to-telugu-and-malayalam-deb5d23e9264).

- Tamil LLaMA v0.2 models are out. It is a significant upgrade compared to the earlier version.
   - Tamil LLaMA is now bilingual and can respond fluently in both English and Tamil.
   - Better tokenizer.
   - Better base model.
   - Better fine-tuning dataset and performance.
   - Our models match or better the performance of Meta's LLaMA 2 in almost all the benchmarks.
- The first-ever Telugu and Malayalam LLaMA models have also been released following the same methodology.

## Table of Contents


- [Available Models](#available-models)
- [Benchmark Scores](#benchmark-scores)
- [Demo](#demo)
- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Prompting Format](#prompting-format-for-instruction-models)
- [Usage Note](#usage-note)
- [Contributions](#contributions)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Available Models

| Model                    | Type                        | Data              | Base Model           | # Params | Download Links                                                         |
|--------------------------|-----------------------------|-------------------|----------------------|------|------------------------------------------------------------------------|
| Tamil LLaMA 7B Base      | Base model                  | 12GB              | LLaMA 7B             | 7B   | [HF Hub](https://huggingface.co/abhinand/tamil-llama-7b-base-v0.1)     |
| Tamil LLaMA 13B Base     | Base model                  | 4GB               | LLaMA 13B            | 13B  | [HF Hub](https://huggingface.co/abhinand/tamil-llama-13b-base-v0.1)    |
| Tamil LLaMA 7B Instruct  | Instruction following model | 145k instructions | Tamil LLaMA 7B Base  | 7B   | [HF Hub](https://huggingface.co/abhinand/tamil-llama-7b-instruct-v0.1) |
| Tamil LLaMA 13B Instruct | Instruction following model | 145k instructions | Tamil LLaMA 13B Base | 13B  | [HF Hub](abhinand/tamil-llama-13b-instruct-v0.1)                       |

### Quantized Version of Available Models

| Model                    | Format | Bits                 | Download Links                                                               |
|--------------------------|--------|----------------------|------------------------------------------------------------------------------|
| Tamil LLaMA 7B Base      | GGUF   | Q4_K_M, Q5_K_M, Q8_0 | [HF Hub](https://huggingface.co/abhinand/tamil-llama-7b-base-v0.1-gguf)      |
| Tamil LLaMA 13B Base     | GGUF   | Q4_K_M, Q5_K_M, Q8_0 | [HF Hub](https://huggingface.co/abhinand/tamil-llama-13b-base-v0.1-gguf)     |
| Tamil LLaMA 7B Instruct  | GGUF   | Q4_K_M, Q5_K_M, Q8_0 | [HF Hub](https://huggingface.co/abhinand/tamil-llama-7b-instruct-v0.1-gguf)  |
| Tamil LLaMA 13B Instruct | GGUF   | Q4_K_M, Q5_K_M, Q8_0 | [HF Hub](https://huggingface.co/abhinand/tamil-llama-13b-instruct-v0.1-gguf) |

## Benchmark Scores

Scores are calculated using the HuggingFace [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

> **Note:** The benchmarks test the model's capabilities in English reasoning; although the Tamil LLaMA models were not trained on quality reasoning tasks in English, they show decent performance across most benchmarks.

| Model                    | Average | ARC   | HellaSwag | MMLU  | TruthfulQA | Winogrande | GSM8K |
|--------------------------|---------|-------|-----------|-------|------------|------------|-------|
| Tamil LLaMA 13B Instruct | **51.59**   | **54.52** | 79.35     | 50.37 | 41.22      | **76.56**      | **7.51**  |
| Tamil LLaMA 13B Base     | 49.5    | 52.82 | **79.95**     | **52.05** | 36.56      | 75.61      | 0     |
| Tamil LLaMA 7B Instruct  | 45.52   | 48.04 | 70.97     | 39.95 | **41.7**       | 70.64      | 1.82  |
| Tamil LLaMA 7B Base      | 44.52   | 46.67 | 72.85     | 40.95 | 35.93      | 70.72      | 0     |

## Demo

> **Update:** There is now a Google Colab demo for Tamil/Telugu/Malayalam LLaMAs part of this project. [Click Here](https://colab.research.google.com/drive/11_RHZim_HubD2NskxSwq4X_NW4XlnrtS?usp=sharing) to open the Colab Notebook.

A simple interactive demo of Tamil-LLaMA-7B-Instruct-v0.1 is hosted in the HuggingFace Space here -> [abhinand/tamil-llama-playground](https://huggingface.co/spaces/abhinand/tamil-llama-playground)

<img src="assets/demo_screenshot.png" alt="Tamil LLaMA Image" width="75%" height="auto">

## Getting Started

### Using LMStudio:

[LM Studio](https://lmstudio.ai/) is an easy-to-use and powerful local GUI for Windows and macOS (Silicon) with GPU acceleration. Linux is available, and it is in beta as of 27/11/2023.

1. **Download and Install LM Studio**: Download LM Studio from the official website.

2. **Locate the Tamil Llama Model**: After installation, open LM Studio and use the search bar to find the "Tamil Llama" model. If you have the GGUF model ID, you can paste it directly into the search bar.

3. **Download the Appropriate Model Variant**: Select the appropriate variant of the Tamil Llama model depending on your system's specifications. Click on the 'Download' button to start the download process.

4. **Import the Preset JSON File**: Once the model is downloaded, navigate to the 'Chat' tab in LM Studio. In the settings, find the 'Preset' menu and click on the dropdown. Select "Import Preset From File" and import the preset JSON file located at [config/lm_studio/model_config.json](config/lm_studio/model_config.json) in the repository.

5. **Select and Load the Model**: Click "Select a model to load" on the top bar. You can choose the Tamil Llama variant you downloaded from the list.

6. **Initiate Conversations with the Model**: The Tamil Llama model is now ready. You can start engaging in conversations in the chat area of LM Studio.

### Using with Ollama:

1. **Verify Ollama Installation**: First, ensure that [Ollama](https://github.com/jmorganca/ollama) is correctly installed on your system. If not, install it from the official source.

2. **Download the Modelfile**: Access the GitHub repository and download the [Modelfile](config/ollama/Modelfile). This file is necessary for setting up the Tamil Llama model in Ollama.

3. **Prepare the Working Directory**: Place the downloaded `Modelfile` and the model's GGUF file in the same directory. To work in this directory, use the `cd` command in your terminal to change to the appropriate directory.

4. **Download the Tamil Llama Model**: Execute the following command in your terminal to download the desired Tamil Llama model from the GitHub repository:

   ```bash
   curl -L https://huggingface.co/abhinand/tamil-llama-7b-instruct-v0.1-gguf/resolve/main/tamil-llama-7b-v0.1-q8_0.gguf -o tamil-llama.gguf
   ```

   This command downloads the Tamil Llama model GGUF file and saves it as `tamil-llama.gguf` in your current directory.

5. **Import and Run the Model in Ollama**: After downloading the model, use the following command to create and run the Tamil Llama model in Ollama:

   ```bash
   ollama create tamil-llama -f Modelfile 
   ```

   This command imports the Tamil Llama model into Ollama and prepares it for use. 

Optionally, depending upon your system's capabilities, make sure to configure these parameters in the Modelfile too:

```
PARAMETER num_thread 8
PARAMETER num_gpu 0
```

For more information regarding the Modelfile's available parameters, check out the [official docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md).

## Datasets

The repository includes a Tamil-translated version of the Alpaca dataset and a subset of the OpenOrca dataset, which are used for instruction fine-tuning and evaluation.

**Tamil Alpaca**: [abhinand/tamil-alpaca](https://huggingface.co/datasets/abhinand/tamil-alpaca)

**Tamil Alpaca Orca**: [abhinand/tamil-alpaca-orca](https://huggingface.co/datasets/abhinand/tamil-alpaca-orca)

**Tamil LLaMA Eval**: [abhinand/tamil-llama-eval](https://huggingface.co/datasets/abhinand/tamil-llama-eval)

## Prompting Format for Instruction Models

**Prompt Template Without Input**

```
{system_prompt}

### Instruction:
{instruction or query}

### Response:
{response}
```

**Prompt Template With Input**

```
{system_prompt}

### Instruction:
{instruction or query}

### Input:
{input}

### Response:
{response}
```

## Usage Note

It's important to note that the models have not undergone detoxification. Therefore, while they possess impressive linguistic capabilities, they could generate content that could be deemed harmful or offensive. We urge users to exercise discretion and closely supervise the model's outputs, especially in public or sensitive applications.

## Contributions

We welcome contributions to this project. If you have suggestions or improvements, please open an issue or a pull request.

## License

This project is licensed under the GNU GPL v3.0 license - see the [LICENSE.md](LICENSE) file for details.

> **IMPORTANT**: The [GPL 3.0 License] (LICENSE) applies solely to the source code and datasets provided. As this project is a derivative of Meta's LLaMA 2 model, it is subject to the original licensing of LLaMA 2, which cannot be altered. So, for comprehensive details regarding the model's licensing, please look at the [LLAMA2-LICENSE](LLAMA2-LICENSE) file.

## Citation

If you use this model or the Tamil-Llama dataset in your research, please cite:

```BibTeX
@misc{balachandran2023tamilllama,
      title={Tamil-Llama: A New Tamil Language Model Based on Llama 2}, 
      author={Abhinand Balachandran},
      year={2023},
      eprint={2311.05845},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact

If you have any questions about the codebase or research, please contact Abhinand Balachandran at abhinandb.ml@gmail.com.
