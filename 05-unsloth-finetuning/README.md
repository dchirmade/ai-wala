# Unsloth AI Fine-Tuning Setup

Professional setup for training Large Language Models (LLMs) using **Unsloth**, optimized for Ubuntu systems with NVIDIA GPUs (tested on RTX 3050 8GB).

## Features
- **Memory Optimized**: Uses 4-bit quantization and gradient checkpointing to fit training on 8GB VRAM.
- **Fast Training**: Leverages Unsloth's optimized kernels for up to 2x faster training and inference.
- **Ready-to-use**: Includes automated environment setup and a demo training script.

## Getting Started

### 1. Requirements
- Ubuntu 20.04+ (22.04 recommended)
- NVIDIA GPU with 8GB+ VRAM
- CUDA Drivers installed

### 2. Installation
Run the setup script to create a virtual environment and install all dependencies:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### 3. Usage
Activate the environment and run the training script:
```bash
source unsloth_env/bin/activate
python train_unsloth.py
```

## Project Structure
- `setup_env.sh`: Automated environment configuration.
- `train_unsloth.py`: Core fine-tuning logic using Llama-3.2-3B.
- `lora_model/`: Default directory for saved LoRA adapters.
- `outputs/`: Training checkpoints and logs.

## VRAM Optimization Tips
- **Reduce `max_seq_length`**: If you hit OOM (Out of Memory), lower this from 2048 to 1024 or 512.
- **Increase `gradient_accumulation_steps`**: Instead of increasing batch size (which uses VRAM), increase accumulation steps to simulate a larger batch.
- **Use `adamw_8bit`**: The script uses 8-bit Adam to save significant memory during training.
