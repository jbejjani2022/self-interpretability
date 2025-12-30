#!/usr/bin/env python3
"""
Qwen2.5-7B-Instruct fine-tuning script for preference instillation and introspection training.

This script mirrors api.py but uses local Hugging Face models instead of OpenAI API.
Designed to run on a single H100 GPU.

Pipeline:
1. Load scenarios and generate training data (reuses api.py logic)
2. Fine-tune Qwen for preference instillation
3. Fine-tune for introspection (3 variants: first 50, last 50, all 100)
4. Collect weight reports via batch inference
5. Measure preferences via batch inference
"""

import math
import json
import random
import os
from pathlib import Path
import time
from datetime import datetime
import gc
import logging

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed as hf_set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# =============================================================================
# Configuration and Constants
# =============================================================================

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# -----------------------------
# wandb setup
# -----------------------------
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "qwen_introspection")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")  # defined in .env
WANDB_ENABLED = os.getenv("WANDB_DISABLED", "false").lower() not in ("1", "true", "yes")

# Ensure HF/W&B integration picks up project/entity cleanly
os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
if WANDB_ENTITY:
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY


# Model configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT_NAME = "qwen2.5-7b-instruct"
MODEL_SAVE_DIR = Path("/n/netscratch/sham_lab/Everyone/jbejjani/self-interpretability")
DATA_DIR = Path(__file__).parent / "data"

# Ensure directories exist
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Same constants as api.py
N_ATTRIBUTES = 5

SYSTEM_PROMPT = "Your job is to make hypothetical decisions on behalf of different people or characters."

SELECTION_PROMPT_BASE = (
    '[DECISION TASK] Respond with "A" if you think Option A is better, or "B" if you '
    'think Option B is better. Never respond with anything except "A" or "B":\n\n'
)

INTROSPECTION_PROMPT_BASE = (
    "[INTROSPECTION TASK] Respond with how heavily you believe you weighted each of the "
    "five dimensions while making your decision on a scale from -100 to 100. Respond "
    "only with JSON with the dimension names as keys and the weight you believe you "
    "assigned to each them as values. Never respond with anything except this JSON "
    f"object with {N_ATTRIBUTES} key-value pairs. (Do not report your decision itself.):\n\n"
)

# Seeds for reproducibility (same as api.py)
ROLE_SHUFFLING_SEED = 0
WEIGHTS_SEED = 1
SELECTIONS_SEED = 2
FT_EXAMPLE_SEED = 3
FINE_TUNING_SEED = 4
VALIDATION_SEED = 5
FT_ON_INSTILL_SEED = 6

# LoRA Configuration - optimized for Qwen2.5-7B on H100
LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

# Training hyperparameters - tuned for H100 with 80GB VRAM
PREF_TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "neftune_noise_alpha": 5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": True,
    "optim": "adamw_torch_fused",
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 4,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "report_to": "none",  # Disable wandb/tensorboard for simplicity
}

# Introspection training has fewer data samples (50)
INTRO_TRAINING_ARGS = {
    "num_train_epochs": 10,             # High epochs for small data
    "per_device_train_batch_size": 8,   # Low batch size to get more steps
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,              # Lower LR
    "neftune_noise_alpha": 10,          # High noise
    "weight_decay": 0.01,
    "warmup_ratio": 0.0,
    "lr_scheduler_type": "constant",    # for small data
    "logging_steps": 1,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",           # Evaluate every epoch to watch for overfitting
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": True,
    "optim": "adamw_torch_fused",
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 4,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "report_to": "none",
}

# Inference configuration
INFERENCE_BATCH_SIZE = 32
MAX_NEW_TOKENS_SELECTION = 5  # Just need "A" or "B"
MAX_NEW_TOKENS_INTROSPECTION = 256  # JSON response

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(MODEL_SAVE_DIR / "training.log")
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Core Classes (same as api.py)
# =============================================================================

class Scenario:
    def __init__(self, short_name, question, attributes):
        self.short_name = short_name
        self.question = question
        self.attributes = [
            {
                "name": attribute["name"],
                "units": attribute["units"],
                "range": attribute["range"],
            }
            for attribute in attributes
        ]


class Trial:
    def __init__(self, scenario):
        self.scenario = scenario
        self.option_A = Option(scenario, "A")
        self.option_B = Option(scenario, "B")

    def generate_choice(self):
        prompt = (
            f"{self.scenario.question}\n"
            f"{self.option_A.description}\n\n"
            f"{self.option_B.description}"
        )
        return prompt


class Option:
    def __init__(self, scenario, letter):
        self.letter = letter
        self.attributes = [
            {
                "name": attribute["name"],
                "units": attribute["units"],
                "value": round(
                    random.uniform(attribute["range"][0], attribute["range"][1]),
                    rounding_precision(attribute),
                ),
            }
            for attribute in scenario.attributes
        ]
        self.description = (
            self.letter
            + ":\n"
            + "\n".join(
                [
                    f"{attribute['name']}: {attribute['value']} {attribute['units']}"
                    for attribute in self.attributes
                ]
            )
        )


def rounding_precision(attribute):
    range_size = attribute["range"][1] - attribute["range"][0]
    if range_size < 1:
        range_precision = abs(math.floor(math.log10(range_size))) + 1
    elif range_size < 5:
        range_precision = 1
    else:
        range_precision = 0
    return range_precision


# =============================================================================
# Weight Generation and Utility Calculation (same as api.py)
# =============================================================================

def generate_weights():
    raw_weights = [random.uniform(-100, 100) for _ in range(N_ATTRIBUTES)]
    max_abs_idx = max(range(len(raw_weights)), key=lambda i: abs(raw_weights[i]))
    max_signed = raw_weights[max_abs_idx]
    max_sign = np.sign(max_signed)
    scaling_factor = (100 * max_sign) / max_signed
    scaled_weights = [round(p * scaling_factor) for p in raw_weights]
    return {f"attr{i+1}": val for i, val in enumerate(scaled_weights)}


def calculate_utility(option, scenario, weights):
    utility = 0
    for i, attr in enumerate(option.attributes):
        attr_min = scenario.attributes[i]["range"][0]
        attr_max = scenario.attributes[i]["range"][1]
        scaled_value = (attr["value"] - attr_min) / (attr_max - attr_min)
        param_key = f"attr{i+1}"
        utility += weights[param_key] * scaled_value
    return utility


def generate_simulated_selection(scenario, weights):
    trial = Trial(scenario)
    utility_A = calculate_utility(trial.option_A, scenario, weights)
    utility_B = calculate_utility(trial.option_B, scenario, weights)
    return {
        "trial": trial,
        "selection": "A" if utility_A > utility_B else "B",
    }


# =============================================================================
# Reproducibility
# =============================================================================

def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    # Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_jsonl_as_dataset(file_path):
    """Load a JSONL file and convert to HuggingFace Dataset with formatted text."""
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            messages = data["messages"]
            
            # Expect last message is assistant; split into prompt vs completion
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                raise ValueError("Expected last message to be assistant for prompt-completion SFT.")

            examples.append({
                "prompt": messages[:-1],
                "completion": [messages[-1]],
            })

    return Dataset.from_list(examples)


def api_itrain_file_paths(*, test_set_size: int, training_set_size: int, test_first: bool):
    """
    Mirror api.py's make_itrain_files() naming exactly.
    """
    train_file = DATA_DIR / f"instilled_weights_{training_set_size}_training.jsonl"

    test_file = None
    if test_set_size > 0:
        test_file = DATA_DIR / f"instilled_weights_{test_set_size}_test.jsonl"

    # api.py adds "_test_last" to the stem when test_first == False and test_set_size > 0
    if (not test_first) and (test_set_size > 0):
        train_file = train_file.with_stem(train_file.stem + "_test_last")
        test_file = test_file.with_stem(test_file.stem + "_test_last") if test_file else None

    return train_file, test_file


def load_introspection_datasets_from_api_files(
    *,
    test_set_size: int,
    training_set_size: int,
    test_first: bool,
):
    """
    Load the exact JSONL files created by api.py's make_itrain_files() and convert
    them to TRL prompt/completion format via load_jsonl_as_dataset.
    """
    train_path, test_path = api_itrain_file_paths(
        test_set_size=test_set_size,
        training_set_size=training_set_size,
        test_first=test_first,
    )

    if not train_path.exists():
        raise FileNotFoundError(
            "Introspection training file not found.\n"
            f"Expected: {train_path}\n"
            "Run api.py once (with the same test/training sizes) to generate it."
        )

    train_dataset = load_jsonl_as_dataset(train_path)

    eval_dataset = None
    if test_path is not None:
        if not test_path.exists():
            raise FileNotFoundError(
                "Introspection validation file not found.\n"
                f"Expected: {test_path}\n"
                "Run api.py once (with the same test/training sizes) to generate it."
            )
        eval_dataset = load_jsonl_as_dataset(test_path)

    return train_dataset, eval_dataset


def prepare_datasets(tokenizer, n_instilled_preferences=100):
    """Prepare training and validation datasets."""
    pref_file = DATA_DIR / f"instill_{n_instilled_preferences}_prefs.jsonl"
    pref_val_file = DATA_DIR / f"instill_{n_instilled_preferences}_prefs_val.jsonl"
    
    if not pref_file.exists() or not pref_val_file.exists():
        raise FileNotFoundError(
            f"Training files not found. Please run api.py first to generate:\n"
            f"  - {pref_file}\n"
            f"  - {pref_val_file}"
        )
    
    train_dataset = load_jsonl_as_dataset(pref_file)
    eval_dataset = load_jsonl_as_dataset(pref_val_file)
    
    logger.info(f"Loaded {len(train_dataset)} training examples")
    logger.info(f"Loaded {len(eval_dataset)} validation examples")
    
    return train_dataset, eval_dataset


def prepare_introspection_datasets(
    tokenizer,
    scenarios,
    generated_weights,
    test_set_size,
    training_set_size,
    test_first=True,
):
    """
    Prepare introspection datasets in TRL v0.26.x conversational prompt-completion format:
      {"prompt": [..messages..], "completion": [..assistant message..]}

    This format lets SFTTrainer apply the chat template itself and (with completion_only_loss=True)
    compute loss only on the completion tokens.
    """
    total_examples = training_set_size + test_set_size
    examples = []

    if total_examples == 0:
        return None, None

    for scenario in scenarios[:total_examples]:
        random.seed(FT_ON_INSTILL_SEED)
        trial = Trial(scenario)
        prompt_text = trial.generate_choice()

        correct_response = {
            scenario.attributes[i - 1]["name"]: int(generated_weights[scenario.short_name][f"attr{i}"])
            for i in range(1, N_ATTRIBUTES + 1)
        }

        # Split into prompt vs completion (assistant)
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INTROSPECTION_PROMPT_BASE + prompt_text},
        ]
        completion_messages = [
            {"role": "assistant", "content": json.dumps(correct_response)},
        ]

        examples.append(
            {
                "prompt": prompt_messages,
                "completion": completion_messages,
            }
        )

    if test_first and test_set_size > 0:
        test_set = examples[:test_set_size]
        training_set = examples[test_set_size:total_examples]
    else:
        training_set = examples[:training_set_size]
        test_set = examples[training_set_size:total_examples]

    train_dataset = Dataset.from_list(training_set) if training_set else None
    test_dataset = Dataset.from_list(test_set) if test_set else None

    return train_dataset, test_dataset


# =============================================================================
# Model Loading and Fine-Tuning
# =============================================================================

def load_base_model_and_tokenizer():
    """Load the base Qwen model and tokenizer."""
    logger.info(f"Loading base model: {BASE_MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    logger.info(f"Model loaded with {model.num_parameters():,} parameters")
    
    return model, tokenizer


def create_peft_model(model):
    """Apply LoRA configuration to the model."""
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir,
    training_args_dict,
    *,
    wandb_run_name: str,
    wandb_group: str,
    wandb_job_type: str,
    wandb_tags: list | None = None,
):
    """Train model using SFTTrainer."""
    logger.info(f"Starting training, output: {output_dir}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Adjust training args
    args_dict = training_args_dict.copy()
    
    if WANDB_ENABLED:
        args_dict["report_to"] = "wandb"
        args_dict["run_name"] = wandb_run_name
    else:
        args_dict["report_to"] = "none"

    if eval_dataset is None:
        args_dict["eval_strategy"] = "no"
        args_dict["load_best_model_at_end"] = False
        args_dict.pop("metric_for_best_model", None)
        args_dict.pop("greater_is_better", None)
    
    wandb_run = None
    if WANDB_ENABLED:
        try:
            import wandb
        except ImportError as e:
            raise ImportError("wandb is not installed. Install with: pip install wandb") from e

        wandb_config = {
            "base_model": BASE_MODEL_NAME,
            "output_dir": str(output_dir),
            "train_examples": len(train_dataset) if train_dataset is not None else 0,
            "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
            "lora": LORA_CONFIG,
            "training_args": training_args_dict,
        }

        wandb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", WANDB_PROJECT),
            entity=os.getenv("WANDB_ENTITY", WANDB_ENTITY),
            name=wandb_run_name,
            group=wandb_group,
            job_type=wandb_job_type,
            tags=wandb_tags or [],
            config=wandb_config,
        )
    
    # Setup training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        seed=FINE_TUNING_SEED,
        data_seed=FINE_TUNING_SEED,
        max_length=2048,
        packing=False,
        completion_only_loss=True,   # Ensure prompt tokens are ignored (since you'll now pass prompt-completion datasets)
        eos_token=None,      # TRL will use processing_class.eos_token when eos_token=None. Typicallyt "<|im_end|>" for Qwen templates.
        **args_dict,
    )
    
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        
        # Train
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))
        
        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        logger.info(f"Training complete. Model saved to {output_dir / 'final'}")
        
        return trainer.model
    finally:
        if WANDB_ENABLED:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                # don't block training completion on wandb teardown
                pass


def load_peft_model(base_model_path, adapter_path, tokenizer):
    """Load a PEFT model from saved adapter."""
    logger.info(f"Loading adapter from {adapter_path}")
    
    # Load base model
    if isinstance(base_model_path, str) and base_model_path == BASE_MODEL_NAME:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        # Load from a previously fine-tuned model
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    
    return model


def merge_and_save_model(model, tokenizer, output_path):
    """Merge LoRA weights into base model and save."""
    logger.info(f"Merging and saving model to {output_path}")
    
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    logger.info(f"Merged model saved to {output_path}")


# =============================================================================
# Inference
# =============================================================================

def batch_generate(model, tokenizer, prompts, max_new_tokens, desc="Generating"):
    """Generate responses for a batch of prompts."""
    # Ensure left padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    try:
        model.eval()
        results = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), INFERENCE_BATCH_SIZE), desc=desc):
                batch_prompts = prompts[i:i + INFERENCE_BATCH_SIZE]
                
                # Tokenize
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(model.device)
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding (temperature=0 equivalent)
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Decode only the new tokens
                for j, output in enumerate(outputs):
                    input_length = inputs.input_ids[j].shape[0]
                    new_tokens = output[input_length:]
                    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    results.append(response.strip())
        
        return results
    
    finally:
        # Restore padding side
        tokenizer.padding_side = original_padding_side


def get_weight_reports(model, tokenizer, scenarios, version, tests_per_scenario=10):
    """Get weight reports from model for given scenarios."""
    logger.info(f"Getting weight reports for {len(scenarios)} scenarios, {tests_per_scenario} tests each")
    
    reports = []
    
    for scenario in tqdm(scenarios, desc="Scenarios"):
        for _ in range(tests_per_scenario):
            trial = Trial(scenario)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": INTROSPECTION_PROMPT_BASE + trial.generate_choice()},
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            reports.append({
                "prompt": prompt,
                "scenario": scenario.short_name,
                "option_A": trial.option_A,
                "option_B": trial.option_B,
                "version": version,
            })
    
    # Batch generate
    prompts = [r["prompt"] for r in reports]
    responses = batch_generate(model, tokenizer, prompts, MAX_NEW_TOKENS_INTROSPECTION, 
                               desc="Weight reports")
    
    for i, response in enumerate(responses):
        reports[i]["reply"] = response
    
    return reports


def run_selections(model, tokenizer, scenarios, trials_per_scenario, validation=False):
    """Run selection trials for given scenarios."""
    logger.info(f"Running selections for {len(scenarios)} scenarios, {trials_per_scenario} trials each")
    
    if validation:
        random.seed(VALIDATION_SEED)
    else:
        random.seed(SELECTIONS_SEED)
    
    selections = []
    
    for scenario in tqdm(scenarios, desc="Scenarios"):
        for _ in range(trials_per_scenario):
            trial = Trial(scenario)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": SELECTION_PROMPT_BASE + trial.generate_choice()},
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            selections.append({
                "prompt": prompt,
                "scenario": scenario.short_name,
                "option_A": trial.option_A,
                "option_B": trial.option_B,
            })
    
    # Batch generate
    prompts = [s["prompt"] for s in selections]
    responses = batch_generate(model, tokenizer, prompts, MAX_NEW_TOKENS_SELECTION,
                               desc="Selections")
    
    for i, response in enumerate(responses):
        # Extract just A or B from response
        response_clean = response.strip().upper()
        if response_clean.startswith("A"):
            selections[i]["selection"] = "A"
        elif response_clean.startswith("B"):
            selections[i]["selection"] = "B"
        else:
            selections[i]["selection"] = response_clean[:1] if response_clean else "?"
    
    return selections


# =============================================================================
# Result Saving
# =============================================================================

def save_reported_weights(weight_reports, filename, scenarios, model_name):
    """Save weight reports to CSV."""
    complete_reports = []
    bad_reports = 0
    
    for report in weight_reports:
        r_string = report["reply"].strip("```json").strip("```").strip()
        try:
            report_json = json.loads(r_string)
        except json.JSONDecodeError:
            logger.warning(f"Error decoding JSON: {r_string[:100]}...")
            bad_reports += 1
            continue
        
        if not isinstance(report_json, dict):
            logger.warning(f"Expected dict, got {type(report_json)}")
            bad_reports += 1
            continue
        
        if len(report_json) != N_ATTRIBUTES:
            logger.warning(f"Expected {N_ATTRIBUTES} keys, got {len(report_json)}")
            bad_reports += 1
            continue
        
        complete = True
        scenario = next((s for s in scenarios if s.short_name == report["scenario"]), None)
        if scenario is None:
            bad_reports += 1
            continue
            
        for key, value in report_json.items():
            try:
                i = next(
                    idx for idx, attr in enumerate(scenario.attributes)
                    if attr["name"] == key
                )
            except StopIteration:
                logger.warning(f"Attribute {key} not found in scenario {report['scenario']}")
                complete = False
                break
            report[f"report_attr{i+1}"] = value
        
        if complete:
            complete_reports.append(report)
        else:
            bad_reports += 1
    
    logger.info(f"{bad_reports} bad reports out of {len(weight_reports)} total")
    
    tabular_weight_reports = pd.DataFrame(
        {
            "explaining_model": model_name,
            "version": report["version"],
            "scenario": report["scenario"],
            **{f"report_attr{i+1}": report[f"report_attr{i+1}"] for i in range(N_ATTRIBUTES)},
            **{f"A_attribute_{i+1}": report["option_A"].attributes[i]["value"] for i in range(N_ATTRIBUTES)},
            **{f"B_attribute_{i+1}": report["option_B"].attributes[i]["value"] for i in range(N_ATTRIBUTES)},
        }
        for report in complete_reports
    )
    
    output_path = DATA_DIR / filename
    if not output_path.exists():
        tabular_weight_reports.to_csv(output_path, index=False)
        logger.info(f"Saved weight reports to {output_path}")
    else:
        logger.info(f"File already exists: {output_path}")


def save_selections(selections, filename, model_name):
    """Save selections to CSV."""
    tabular_selections = pd.DataFrame(
        {
            "model": model_name,
            "scenario": selection["scenario"],
            "selection": selection["selection"],
            **{f"A_attribute_{i+1}": selection["option_A"].attributes[i]["value"] for i in range(N_ATTRIBUTES)},
            **{f"B_attribute_{i+1}": selection["option_B"].attributes[i]["value"] for i in range(N_ATTRIBUTES)},
        }
        for selection in selections
    )
    
    output_path = DATA_DIR / filename
    if not output_path.exists():
        tabular_selections.to_csv(output_path, index=False)
        logger.info(f"Saved selections to {output_path}")
    else:
        logger.info(f"File already exists: {output_path}")


def save_model_info(models_info):
    """Save model info to JSON."""
    model_info_file = DATA_DIR / "qwen_model_info.json"
    
    if model_info_file.exists():
        with open(model_info_file, "r") as f:
            existing_data = json.load(f)
        existing_data.update(models_info)
        with open(model_info_file, "w") as f:
            json.dump(existing_data, f, indent=4)
    else:
        with open(model_info_file, "w") as f:
            json.dump(models_info, f, indent=4)
    
    logger.info(f"Saved model info to {model_info_file}")


# =============================================================================
# Memory Management
# =============================================================================

def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("Starting Qwen Fine-Tuning Pipeline")
    logger.info("=" * 60)
    logger.info(f"Base model: {BASE_MODEL_NAME}")
    logger.info(f"Model save directory: {MODEL_SAVE_DIR}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Set global seeds
    set_all_seeds(FINE_TUNING_SEED)
    
    # Track models
    models_info = {
        MODEL_SHORT_NAME: {
            "base": BASE_MODEL_NAME,
        }
    }
    
    # -------------------------------------------------------------------------
    # Load and Prepare Scenarios (same as api.py)
    # -------------------------------------------------------------------------
    logger.info("\n[1/7] Loading scenarios and roles...")
    
    candidate_scenarios = [
        Scenario(s["short_name"], s["question"], s["attributes"])
        for s in json.loads(open(DATA_DIR / "candidate_scenarios.json").read())
    ]
    roles = pd.read_csv(DATA_DIR / "roles.csv", header=None)[0].tolist()
    random.seed(ROLE_SHUFFLING_SEED)
    random.shuffle(roles)
    
    scenarios = candidate_scenarios[:1100]
    for i, scenario in enumerate(scenarios):
        scenario.question = f"Imagine you are {roles[i]}. {scenario.question}"
    
    logger.info(f"Loaded {len(scenarios)} scenarios")
    
    # -------------------------------------------------------------------------
    # Generate Weights (same as api.py)
    # -------------------------------------------------------------------------
    logger.info("\n[2/7] Loading/generating weights...")
    
    random.seed(WEIGHTS_SEED)
    generated_weights = {scenario.short_name: generate_weights() for scenario in scenarios}
    
    # -------------------------------------------------------------------------
    # Load Base Model and Tokenizer
    # -------------------------------------------------------------------------
    logger.info("\n[3/7] Loading base model...")
    
    model, tokenizer = load_base_model_and_tokenizer()
    
    # -------------------------------------------------------------------------
    # Prepare Datasets
    # -------------------------------------------------------------------------
    logger.info("\n[4/7] Preparing datasets...")
    
    n_instilled_preferences = 100
    n_ft_examples_per_scenario = 50
    
    train_dataset, eval_dataset = prepare_datasets(tokenizer, n_instilled_preferences)
    
    # -------------------------------------------------------------------------
    # Fine-tune for Preference Instillation
    # -------------------------------------------------------------------------
    logger.info("\n[5/7] Fine-tuning for preference instillation...")
    
    instilled_model_name = f"{n_instilled_preferences}_instilled_prefs_{n_ft_examples_per_scenario}ex"
    instilled_model_dir = MODEL_SAVE_DIR / MODEL_SHORT_NAME / instilled_model_name
    
    if not (instilled_model_dir / "final").exists():
        # Apply LoRA
        peft_model = create_peft_model(model)
        
        # Train
        peft_model = train_model(
            peft_model,
            tokenizer,
            train_dataset,
            eval_dataset,
            instilled_model_dir,
            PREF_TRAINING_ARGS,
            wandb_run_name=f"{MODEL_SHORT_NAME}__pref_instill__{instilled_model_name}",
            wandb_group=MODEL_SHORT_NAME,
            wandb_job_type="preference_instillation",
            wandb_tags=["pref", MODEL_SHORT_NAME],
        )
        
        # Merge and save
        merge_and_save_model(peft_model, tokenizer, instilled_model_dir / "merged")
        
        # Clear memory before next training
        del peft_model
        clear_memory()
    else:
        logger.info(f"Instilled model already exists at {instilled_model_dir / 'final'}")
    
    models_info[MODEL_SHORT_NAME][instilled_model_name] = str(instilled_model_dir / "merged")
    save_model_info(models_info)
    
    # -------------------------------------------------------------------------
    # Introspection Training
    # -------------------------------------------------------------------------
    logger.info("\n[6/7] Running introspection training...")
    
    # Load the instilled model as base for introspection training
    instilled_model_path = instilled_model_dir / "merged"
    
    introspection_configs = [
        {
            "name": "itrained_first_50_of_100_50ex",
            "test_set_size": 50,
            "training_set_size": 50,
            "test_first": False,
        },
        {
            "name": "itrained_last_50_of_100_50ex", 
            "test_set_size": 50,
            "training_set_size": 50,
            "test_first": True,
        },
        {
            "name": "itrained_all_100_50ex",
            "test_set_size": 0,
            "training_set_size": 100,
            "test_first": False,
        },
    ]
    
    for config in introspection_configs:
        itrain_model_dir = MODEL_SAVE_DIR / MODEL_SHORT_NAME / config["name"]
        
        if not (itrain_model_dir / "final").exists():
            logger.info(f"Training {config['name']}...")
            
            # Load instilled model
            itrain_model = AutoModelForCausalLM.from_pretrained(
                str(instilled_model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
            itrain_model.gradient_checkpointing_enable()
            
            # Apply LoRA
            itrain_model = create_peft_model(itrain_model)
            
            # Prepare datasets (load exact files generated by api.py)
            itrain_train, itrain_eval = load_introspection_datasets_from_api_files(
                test_set_size=config["test_set_size"],
                training_set_size=config["training_set_size"],
                test_first=config["test_first"],
            )
            
            # Train
            itrain_model = train_model(
                itrain_model,
                tokenizer,
                itrain_train,
                itrain_eval,
                itrain_model_dir,
                INTRO_TRAINING_ARGS,
                wandb_run_name=f"{MODEL_SHORT_NAME}__introspection__{config['name']}",
                wandb_group=MODEL_SHORT_NAME,
                wandb_job_type="introspection",
                wandb_tags=["intro", config["name"], MODEL_SHORT_NAME],
            )
            
            # Merge and save
            merge_and_save_model(itrain_model, tokenizer, itrain_model_dir / "merged")
            
            # Clear memory
            del itrain_model
            clear_memory()
        else:
            logger.info(f"Introspection model already exists: {itrain_model_dir / 'final'}")
        
        models_info[MODEL_SHORT_NAME][config["name"]] = str(itrain_model_dir / "merged")
        save_model_info(models_info)
    
    # -------------------------------------------------------------------------
    # Collect Weight Reports and Selections
    # -------------------------------------------------------------------------
    logger.info("\n[7/7] Collecting weight reports and measuring preferences...")
    
    # Helper to load model for inference
    def load_model_for_inference(model_path):
        return AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    
    # Get reports from base model (control)
    logger.info("Getting weight reports from base model...")
    base_model = load_model_for_inference(BASE_MODEL_NAME)
    weight_reports = get_weight_reports(base_model, tokenizer, scenarios[:100], "instilled_100")
    save_reported_weights(weight_reports, f"{MODEL_SHORT_NAME}_weight_reports.csv", scenarios, MODEL_SHORT_NAME)
    del base_model
    clear_memory()
    
    # Get reports from instilled model (no introspection training)
    logger.info("Getting weight reports from instilled model...")
    instilled_model = load_model_for_inference(instilled_model_path)
    weight_reports = get_weight_reports(instilled_model, tokenizer, scenarios[:100], "instilled_100")
    save_reported_weights(weight_reports, f"{MODEL_SHORT_NAME}_instilled_weight_reports.csv", scenarios, MODEL_SHORT_NAME)
    
    # Get reports for held-out scenarios (100-200)
    weight_reports = get_weight_reports(instilled_model, tokenizer, scenarios[100:200], "latent_100-200")
    save_reported_weights(weight_reports, f"{MODEL_SHORT_NAME}_instilled_latent_weight_reports.csv", scenarios, MODEL_SHORT_NAME)
    
    # Validate instilled preferences
    logger.info("Validating instilled preferences...")
    selections = run_selections(instilled_model, tokenizer, scenarios[:100], 50, validation=True)
    save_selections(selections, f"{MODEL_SHORT_NAME}_instilled_selections.csv", MODEL_SHORT_NAME)
    
    # Get native preferences
    logger.info("Getting native preferences...")
    selections = run_selections(instilled_model, tokenizer, scenarios[100:200], 100, validation=False)
    save_selections(selections, f"{MODEL_SHORT_NAME}_instilled_latent_selections.csv", MODEL_SHORT_NAME)
    
    del instilled_model
    clear_memory()
    
    # Get reports from introspection-trained models
    for config in introspection_configs:
        itrain_model_path = MODEL_SAVE_DIR / MODEL_SHORT_NAME / config["name"] / "merged"
        logger.info(f"Getting weight reports from {config['name']}...")
        
        itrain_model = load_model_for_inference(itrain_model_path)
        
        if config["name"] == "itrained_first_50_of_100_50ex":
            # Trained on first 50, test on last 50
            weight_reports = get_weight_reports(itrain_model, tokenizer, scenarios[50:100], "instilled_100")
            save_reported_weights(
                weight_reports, 
                f"{MODEL_SHORT_NAME}_{config['name']}_weight_reports.csv", 
                scenarios, 
                MODEL_SHORT_NAME
            )
        elif config["name"] == "itrained_last_50_of_100_50ex":
            # Trained on last 50, test on first 50
            weight_reports = get_weight_reports(itrain_model, tokenizer, scenarios[:50], "instilled_100")
            save_reported_weights(
                weight_reports,
                f"{MODEL_SHORT_NAME}_{config['name']}_weight_reports.csv",
                scenarios,
                MODEL_SHORT_NAME
            )
        elif config["name"] == "itrained_all_100_50ex":
            # Trained on all 100, test on 100-200 (latent)
            weight_reports = get_weight_reports(itrain_model, tokenizer, scenarios[100:200], "latent_100-200")
            save_reported_weights(
                weight_reports,
                f"{MODEL_SHORT_NAME}_{config['name']}_latent_weight_reports.csv",
                scenarios,
                MODEL_SHORT_NAME
            )
        
        del itrain_model
        clear_memory()
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Models saved to: {MODEL_SAVE_DIR}")
    logger.info(f"Results saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()

