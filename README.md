# Tamil-Llama: A Family of LLaMA-based LLMs focused on Tamil Language

<img src="assets/introducing_tamil_llama.png" alt="Tamil LLaMA Image" width="300" height="auto">

## Description

This repository contains the code and models for "Tamil-Llama", a project focused on enhancing the performance of language models for the Tamil language. It builds upon the open-source LLaMA model, introducing additional Tamil tokens and employing the LoRA methodology for efficient training. Please read the technical report for more details.

Technical Report: [https://arxiv.org/abs/2311.05845](https://arxiv.org/abs/2311.05845)

## Table of Contents


- [Available Models](#available-models)
- [Demo](#demo)
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

## Demo

A simple interactive demo of Tamil-LLaMA-7B-Instruct-v0.1 is hosted in the HuggingFace Space here -> [abhinand/tamil-llama-playground](https://huggingface.co/spaces/abhinand/tamil-llama-playground)

<img src="assets/demo_screenshot.png" alt="Tamil LLaMA Image" width="75%" height="auto">

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

It's important to note that the models have not undergone detoxification. Therefore, while they possess impressive linguistic capabilities, there is a possibility for them to generate content that could be deemed harmful or offensive. We urge users to exercise discretion and supervise the model's outputs closely, especially in public or sensitive applications.

## Contributions

We welcome contributions to this project. If you have suggestions or improvements, please open an issue or a pull request.

## License

This project is licensed under the GNU GPL v3.0 license - see the [LICENSE.md](LICENSE) file for details.

## Citation

If you use this model or the Tamil-Llama dataset in your research, please cite:

```bibtex
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

For any queries regarding the codebase or research, please reach out to Abhinand Balachandran at abhinandb.ml@gmail.com.
