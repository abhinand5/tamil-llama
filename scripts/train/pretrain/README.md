# README for Pretraining Scripts

## Introduction
This README provides essential guidelines for using the pretraining scripts, particularly for Llama 2 pretraining. It addresses specific issues and deprecated logic in the existing scripts and outlines steps to ensure a smooth pretraining process.

## Prerequisites
Before proceeding, ensure you have the following prerequisites:

1. **Custom Transformers Library**: Use the `transformers` library from my clone. This version includes a small but crucial patch that enables resuming pretraining from checkpoints.
    ```bash
    pip install git+https://github.com/abhinand5/transformers.git@abhinand5-deepspeed-patch
    ```

2. **PEFT Library Version**: Install a specific version of the `peft` library. Recent updates in the `peft` packages have introduced compatibility issues, and using the specified version helps avoid these problems.
    ```bash
    pip install git+https://github.com/huggingface/peft.git@13e53fc
    ```

## Guidelines for Llama 2 Pretraining

If you plan to use these scripts for Llama 2 pretraining in your own languages or specifically for Tamil, follow these additional steps:

1. **Delete Safetensors Binaries**: The original Llama model from Hugging Face comes with *safetensors* related binaries. It is crucial to delete these binaries. If not removed, the pretraining script defaults to loading safetensors, which can interfere with saving the modified embedding layer during checkpointing.

   > **Note**: While there is a possibility that my assumption about the safetensors issue might be incorrect, removing these binaries resolved loading issues in a recent pretraining run.

## Upcoming Updates

We are planning to roll out updates to the [run_clm_with_peft.py](./run_clm_with_peft.py) script to address and prevent issues like these. Stay tuned for these enhancements to ensure a more streamlined and error-free pretraining experience.

---

For any further questions or support, please feel free to open an issue in the repository or contact the maintainers directly.

---

Happy Pretraining! 🚀

---