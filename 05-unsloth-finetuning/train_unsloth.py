import logging
import sys
import os

# ==========================================
# 0. Logging Setup (MUST BE FIRST)
# ==========================================
VERBOSE_LEVEL = 15
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

def _verbose(self, message, *args, **kws):
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, message, args, **kws)
logging.Logger.verbose = _verbose

logging.basicConfig(
    level=VERBOSE_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_run.log", mode='a') # Use append mode to be safe
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Now heavy imports
from unsloth import FastLanguageModel
import re
import json
import platform
import time
import threading
import subprocess
import torch
import psutil
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer

# Try importing HfApi for dynamic model fetching
try:
    from huggingface_hub import HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# ==========================================
# 1. System Monitoring & Hardware Utils
# ==========================================
class SystemMonitor:
    """
    Runs in the background to log system resources (CPU, RAM, GPU)
    during the training process.
    """
    def __init__(self, interval=10):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)

    def start(self):
        logger.info("Starting background system monitor...")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        logger.info("Stopping background system monitor...")

    def get_snapshot(self):
        """Returns a string snapshot of current system state."""
        cpu_usage = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_usage = ram.percent
        ram_gb = ram.used / (1024**3)
        
        gpu_stats = "GPU: N/A"
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            
            # Fix for ModuleNotFoundError: nvidia-ml-py
            try:
                # torch.cuda.utilization() requires nvidia-ml-py
                if hasattr(torch.cuda, 'utilization'):
                    gpu_util = torch.cuda.utilization()
                else:
                    gpu_util = 0
            except Exception:
                # Fallback if driver/library issues occur
                gpu_util = 0

            gpu_stats = f"GPU Util: {gpu_util}% | VRAM Alloc: {vram_allocated:.2f}GB | Res: {vram_reserved:.2f}GB"
        
        return f"CPU: {cpu_usage}% | RAM: {ram_usage}% ({ram_gb:.1f}GB) | {gpu_stats}"

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            logger.info(f"FYI [SYS MON] {self.get_snapshot()}")
            time.sleep(self.interval)

def get_detailed_system_info():
    """Prints full hardware specs at startup and returns VRAM in GB."""
    logger.info("="*80)
    logger.info(" SYSTEM HARDWARE DIAGNOSTICS")
    logger.info("="*80)
    
    # CPU
    logger.info(f"OS:      {platform.system()} {platform.release()}")
    logger.info(f"CPU:     {platform.processor()} ({os.cpu_count()} Cores)")
    logger.info(f"RAM:     {psutil.virtual_memory().total / (1024**3):.2f} GB Total")
    
    # GPU
    vram_gb = 0.0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU:     {gpu_name}")
        logger.info(f"VRAM:    {vram_gb:.2f} GB Available")
        logger.info(f"CUDA:    {torch.version.cuda}")
    else:
        logger.info("GPU:     None (CPU Only - Not Supported for Unsloth)")
        
    logger.info("="*80)
    return vram_gb

# ==========================================
# 2. Configuration & Model Selection
# ==========================================
@dataclass
class ModelOption:
    key: str
    name: str
    description: str
    min_vram: float
    is_supported: bool = True
    unsupported_reason: str = ""

class ModelRegistry:
    @staticmethod
    def _estimate_vram(model_name: str) -> float:
        """
        Technical Heuristic to guess 4-bit VRAM requirements + Overhead.
        Base math: Parameters * 0.7-0.8 bytes (4bit) + ~1-2GB CUDA kernels/Activations.
        """
        name = model_name.lower()
        if "1b" in name: return 3.5  # ~1GB model + overhead
        if "3b" in name: return 5.5  # ~2.5GB model + overhead
        if "7b" in name or "8b" in name: return 7.5 # ~5-6GB model + overhead (Fits tightly in 8GB)
        if "9b" in name: return 8.5
        if "11b" in name: return 10.0
        if "13b" in name or "14b" in name: return 12.0
        if "27b" in name: return 20.0
        if "34b" in name: return 24.0
        if "70b" in name or "72b" in name: return 48.0
        return 8.0 # Default fallback

    @staticmethod
    def _generate_description(model_name: str) -> str:
        name = model_name.lower()
        if "llama-3.2" in name: return "Meta's efficient edge model."
        if "llama-3.1" in name: return "Meta's standard open model."
        if "qwen" in name: return "Coding/Math specialist."
        if "mistral" in name: return "High performance generalist."
        if "phi" in name: return "Microsoft high-context model."
        if "gemma" in name: return "Google open model."
        return "Unsloth optimized."

    @staticmethod
    def fetch_options(system_vram: float) -> List[ModelOption]:
        """
        Fetches models and marks them as Supported/Unsupported based on hardware.
        """
        options = []
        
        # Fallback list if API fails
        raw_models = [
            ("unsloth/Llama-3.2-1B-Instruct", "Fallback"),
            ("unsloth/Llama-3.2-3B-Instruct", "Fallback"),
            ("unsloth/Meta-Llama-3.1-8B-Instruct", "Fallback"),
            ("unsloth/Qwen2.5-14B-Instruct", "Fallback"),
            ("unsloth/Llama-3.1-70B-Instruct", "Fallback")
        ]

        if HF_HUB_AVAILABLE:
            logger.verbose("Fetching top models from Hugging Face Hub (unsloth/bnb-4bit)...")
            try:
                api = HfApi()
                # Fetch MORE models (limit 100) to find enough Instruct versions
                # Filter down to top 15 relevant ones.
                models = api.list_models(
                    author="unsloth",
                    search="bnb-4bit", 
                    sort="downloads", 
                    direction=-1, 
                    limit=100
                )
                raw_models = []
                for m in models:
                    # Filter for Instruct/Chat and avoid duplicates or base models
                    if "instruct" in m.modelId.lower() or "chat" in m.modelId.lower():
                        raw_models.append((m.modelId, "Fetched"))
                        if len(raw_models) >= 15: # Stop after finding 15 good candidates
                            break
            except Exception as e:
                logger.error(f"API Fetch failed ({e}). Using fallback list.")

        count = 1
        for model_id, source in raw_models:
            vram_req = ModelRegistry._estimate_vram(model_id)
            desc = ModelRegistry._generate_description(model_id)
            
            # Hardware Suitability Check
            is_supported = True
            reason = ""
            
            # VRAM Check
            if system_vram > 0 and vram_req > system_vram:
                is_supported = False
                reason = f"Requires ~{vram_req}GB VRAM (Have {system_vram:.1f}GB)"
            
            # Apple Silicon Check (Unsloth specific)
            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                is_supported = False
                reason = "Unsloth does not support Apple Silicon/MPS yet."

            # CPU Check
            if not torch.cuda.is_available():
                is_supported = False
                reason = "Unsloth requires NVIDIA GPU (CUDA)."

            options.append(ModelOption(
                key=str(count),
                name=model_id,
                description=desc,
                min_vram=vram_req,
                is_supported=is_supported,
                unsupported_reason=reason
            ))
            count += 1
            
        return options

@dataclass
class TrainingConfig:
    model_name: str = ""
    final_model_name: str = "wsa-bot" # User defined name for the final app
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    learning_rate: float = 2e-4
    max_steps: int = 60
    batch_size: int = 1
    grad_accumulation: int = 4
    output_dir: str = "wsa_model_outputs"
    data_file: str = "data/wsa_knowledge_base.json"

    def apply_hardware_optimizations(self):
        if not torch.cuda.is_available(): return
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Optimization Logic
        if vram <= 8.5:
            self.batch_size = 2
            self.grad_accumulation = 4
            self.max_seq_length = 2048 
        elif vram <= 16.0:
            self.batch_size = 4
            self.grad_accumulation = 2
        else:
            self.batch_size = 8
            self.grad_accumulation = 1

# ==========================================
# 3. Data Management
# ==========================================
class WSADataManager:
    def __init__(self, config: TrainingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def _generate_synthetic_data(self):
        synthetic_data = [
            {"instruction": "What is the mission statement of WS Audiology?", "input": "", "output": "WS Audiology's mission is 'Wonderful Sound for All'."},
            {"instruction": "What are the primary brands under WS Audiology?", "input": "", "output": "Major brands include Widex, Signia, and Rexton."},
            {"instruction": "Explain the difference between Widex and Signia.", "input": "", "output": "Widex focuses on natural sound (ZeroDelay/PureSound). Signia focuses on speech clarity in noise (Integrated Xperience)."},
        ]
        os.makedirs(os.path.dirname(self.config.data_file), exist_ok=True)
        with open(self.config.data_file, 'w') as f:
            json.dump(synthetic_data, f, indent=2)

    def format_prompts(self, examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}""" + self.tokenizer.eos_token
            texts.append(text)
        return { "text" : texts }

    def load_and_prepare(self) -> Dataset:
        if not os.path.exists(self.config.data_file):
            self._generate_synthetic_data()
        dataset = load_dataset("json", data_files=self.config.data_file, split="train")
        return dataset.map(self.format_prompts, batched=True)

# ==========================================
# 4. Main Trainer Class
# ==========================================
class WSATrainer:
    def __init__(self):
        self.monitor = SystemMonitor(interval=5)
        self.config = TrainingConfig()
        self.system_vram = get_detailed_system_info()

    def log_phase(self, phase_name: str, educational_text: str):
        """
        Logs a detailed educational block about the current phase and system state.
        """
        logger.info("\n" + "#"*80)
        logger.info(f" PHASE: {phase_name}")
        logger.info("#"*80)
        logger.info(f"📘 What am I going to do?\n{educational_text}")
        logger.info("-" * 80)
        logger.verbose(f"SYSTEM SNAPSHOT: {self.monitor.get_snapshot()}")
        logger.info("-" * 80)

    def console_selection(self):
        self.log_phase("HARDWARE & MODEL SELECTION", 
                       "1. Scanning Hugging Face: I'm looking for models that are 'quantized' (shrunk) to 4-bit precision.\n"
                       "   - Why? Standard models are too big for most consumer GPUs. 4-bit models use 70% less VRAM.\n"
                       "2. Checking Your Hardware: I'm calculating your GPU's VRAM (Video RAM).\n"
                       "3. Matching: I will compare the model's size against your VRAM. If a model is too big, I'll mark it UNSUPPORTED to prevent crashes.")

        available_options = ModelRegistry.fetch_options(self.system_vram)
        
        # Find best recommended model (Largest one that fits)
        recommended_model = available_options[0].name
        for opt in available_options:
            if opt.is_supported and "3b" in opt.name.lower(): # Prefer 3B for 3050/8GB
                recommended_model = opt.name
                break

        # Since this is interactive, we use print for the selection interface
        print(f"{'#':<3} | {'Model Name':<45} | {'VRAM Req':<9} | {'Status':<12} | {'Reason / Description'}")
        print("-" * 115)
        
        valid_indices = []
        for opt in available_options:
            if opt.is_supported:
                status_icon = "✅ OK"
                reason_text = opt.description
                valid_indices.append(opt.key)
                if opt.name == recommended_model:
                    reason_text += " <== RECOMMENDED"
            else:
                status_icon = "❌ NO"
                reason_text = f"UNSUPPORTED: {opt.unsupported_reason}"

            print(f"{opt.key:<3} | {opt.name:<45} | {opt.min_vram:<4}GB    | {status_icon:<12} | {reason_text}")
            
        print("-" * 115)
        
        choice = input(f"\nEnter selection ID (Default: {recommended_model}): ").strip()
        
        # Validation Logic
        selected_model = recommended_model
        for opt in available_options:
            if choice == opt.key:
                if not opt.is_supported:
                    print(f"\n[WARNING] You selected an UNSUPPORTED model ({opt.name}).")
                    print(f"Reason: {opt.unsupported_reason}")
                    confirm = input("Are you sure you want to proceed? This will likely crash. (y/n): ")
                    if confirm.lower() != 'y':
                        return self.console_selection() # Recursively try again
                selected_model = opt.name
                break
                
        self.config.model_name = selected_model
        logger.info(f"Selected Base Model: {self.config.model_name}")
        self.config.apply_hardware_optimizations()

    def ask_output_name(self):
        print("\n" + "="*80)
        print(" CONFIGURATION: CUSTOM MODEL NAME")
        print("="*80)
        print("Please name your new AI assistant.")
        print("This name will be used to run the model later (e.g., 'ollama run my-wsa-bot').")
        
        default_name = "wsa-bot"
        name = input(f"Enter model name (Press Enter for '{default_name}'): ").strip()
        
        # Sanitize name (Ollama prefers lowercase, alphanumeric, dashes)
        clean_name = re.sub(r'[^a-z0-9\-_]', '', name.lower())
        
        if not clean_name:
            clean_name = default_name
            
        self.config.final_model_name = clean_name
        logger.info(f"✅ Final Model will be created as: '{self.config.final_model_name}'")

    def run(self):
        # 1. Selection
        self.console_selection()
        
        # 2. Output Configuration
        self.ask_output_name()
        
        # 2. Setup
        self.log_phase("MODEL LOADING", 
                       f"1. Downloading: Fetching the architecture and weights for {self.config.model_name}.\n"
                       "2. Quantization (NF4): I'm loading the model in 4-bit 'Normal Float' precision.\n"
                       "   - This keeps the model smart but reduces its memory footprint by ~4x compared to 16-bit.\n"
                       "3. Allocating VRAM: The base model will sit in your GPU memory, but its weights are frozen.")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.config.model_name,
            max_seq_length = self.config.max_seq_length,
            dtype = None,
            load_in_4bit = self.config.load_in_4bit,
        )

        self.log_phase("LORA ADAPTER INJECTION", 
                       "1. Freezing Base Model: The massive Llama/Qwen model will NOT be changed.\n"
                       "2. Injecting LoRA Adapters: I'm adding tiny, trainable matrices (Rank=16) to the attention layers.\n"
                       "   - These adapters act like 'post-it notes' on top of the brain, learning new information without rewriting the whole book.\n"
                       "   - This reduces trainable parameters by ~98%, making training possible on consumer GPUs.")

        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )

        # 4. Data
        self.log_phase("DATASET PREPARATION & TOKENIZATION", 
                       f"1. Loading Data: Reading {self.config.data_file} which contains your custom Q&A pairs.\n"
                       "2. Tokenization: Translating English text into numerical 'Input IDs' that the model understands.\n"
                       "   - Example: 'Hello' -> [123, 456].\n"
                       "3. Formatting: Wrapping your data in the Alpaca template:\n"
                       "   - '### Instruction: ... ### Response: ...'\n"
                       "   - This creates a 'Prompt' that teaches the model how to answer questions.")
        
        data_manager = WSADataManager(self.config, tokenizer)
        dataset = data_manager.load_and_prepare()

        # 5. Training
        self.log_phase("TRAINING LOOP (FINE-TUNING)", 
                       f"1. Starting SFT (Supervised Fine-Tuning): The model will read your valid examples and try to predict the next word.\n"
                       f"2. Backpropagation: If it guesses wrong, we calculate the 'Loss' (error) and adjust the LoRA adapters.\n"
                       f"3. Optimization: Using AdamW 8-bit optimizer to efficiently update the weights.\n"
                       f"   - Batch Size: {self.config.batch_size} (Examples processed at once)\n"
                       f"   - Monitor: Watch the 'loss' go down. Lower is better!")

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = self.config.max_seq_length,
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = self.config.batch_size,
                gradient_accumulation_steps = self.config.grad_accumulation,
                warmup_steps = 5,
                max_steps = self.config.max_steps,
                learning_rate = self.config.learning_rate,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = self.config.output_dir,
                report_to = "none",
            ),
        )

        self.monitor.start() # START MONITORING
        try:
            trainer.train()
        finally:
            self.monitor.stop() # STOP MONITORING
        
        self.log_phase("SAVING ADAPTERS", 
                       "1. Training Complete: The loop is finished.\n"
                       "2. Extraction: I am extracting ONLY the learned LoRA adapters (the 'post-it notes').\n"
                       "3. Saving: Writing these adapters to the 'lora_model' folder.\n"
                       "   - This folder is small (<100MB) but contains all the new intelligence you just taught it.")
        model.save_pretrained("lora_model")
        tokenizer.save_pretrained("lora_model")

        # 5. Inference
        self.log_phase("INFERENCE TEST (VERIFICATION)", 
                       "1. Switching Mode: Putting the model into 'Inference Mode' (runs 2x faster than training).\n"
                       "2. New Test: I will feed it a question it might not have seen exactly, to check if it learned the *concept*.\n"
                       "3. Generation: Watch the output below to see your new AI speak!")
        self.run_inference(model, tokenizer)

        # 6. Export
        self.log_phase("EXPORT & OLLAMA INTEGRATION", 
                       "1. Merging: I am permanently fusing the LoRA adapters into the base model weights.\n"
                       "   - This creates a standalone model that doesn't need Unsloth to run.\n"
                       "2. GGUF Conversion: Converting the model to the .gguf format, which is universal and runs on CPUs/GPUs.\n"
                       "3. Registration: I will run 'ollama create' to register this new model so you can use it anywhere.")
        self.export_model(model, tokenizer)

    def run_inference(self, model, tokenizer):
        FastLanguageModel.for_inference(model)
        prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explain the difference between Widex and Signia.

### Input:


### Response:
"""
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        text_streamer = TextStreamer(tokenizer)
        model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

    def export_model(self, model, tokenizer):
        try:
            # Clean up old export files to avoid conflicts (e.g., between 1B and 3B models)
            import shutil
            for folder in ["model_ollama", "model_ollama_gguf"]:
                if os.path.exists(folder):
                    logger.info(f"Cleaning up old export folder: {folder}")
                    shutil.rmtree(folder)

            # Force float16 to avoid BF16 conversion errors common in GGUF tools on some hardware
            logger.info("Attempting GGUF export...")
            model.save_pretrained_gguf("model_ollama", tokenizer, quantization_method="q4_k_m")
            logger.info("="*50)
            logger.info(" EXPORT SUCCESSFUL")
            logger.info("="*50)

            # --- Automation Logic ---
            logger.info(f"Automating Ollama Registration for '{self.config.final_model_name}'...")
            
            # Unsloth already generates a high-quality Modelfile in the export folder
            modelfile_path = "model_ollama_gguf/Modelfile"
            if not os.path.exists(modelfile_path):
                # Fallback if unsloth changed its behavior
                logger.warning(f"Default Modelfile not found at {modelfile_path}. Checking root...")
                modelfile_path = "Modelfile"

            logger.info(f"Running 'ollama create {self.config.final_model_name}' using {modelfile_path}...")
            subprocess.run(["ollama", "create", self.config.final_model_name, "-f", modelfile_path], check=True) 
            logger.info(f"Model '{self.config.final_model_name}' successfully registered.")

            logger.info("="*50)
            logger.info(f" STARTING CHAT WITH {self.config.final_model_name} (Type /bye to exit)")
            logger.info("="*50)
            subprocess.run(["ollama", "run", self.config.final_model_name])

        except Exception as e:
            logger.error(f"Export/Automation failed: {e}")

if __name__ == "__main__":
    try:
        app = WSATrainer()
        app.run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        logger.exception("Critical Error")