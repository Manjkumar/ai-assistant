import os
import sys
import torch
import jsonlines
from dataclasses import dataclass, field
from typing import Optional, List
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mac_config import mac_config

@dataclass
class ModelArguments:
    model_name: str = field(default="microsoft/phi-2")
    local_model_path: Optional[str] = field(default="/path-to-your-base_model-incase-local")
    use_8bit: bool = field(default=False)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)

@dataclass
class DataArguments:
    train_file: str = field(default="data/retail_training.jsonl")
    val_file: str = field(default="data/retail_validation.jsonl")
    max_length: int = field(default=512)

@dataclass
class TrainingArgs:
    output_dir: str = field(default="models/retail-assistant")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    warmup_steps: int = field(default=100)
    logging_steps: int = field(default=50)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=2)

class RetailAssistantTrainer:
    def __init__(self, model_args, data_args, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.device = mac_config.get_device()

    def load_data(self):
        """Load and prepare dataset"""
        train_data = []
        val_data = []

        # Load training data
        with jsonlines.open(self.data_args.train_file) as reader:
            for obj in reader:
                text = f"""### Instruction: {obj['instruction']}
### Input: {obj['input']}
### Response: {obj['output']}"""
                train_data.append({"text": text})

        # Load validation data
        with jsonlines.open(self.data_args.val_file) as reader:
            for obj in reader:
                text = f"""### Instruction: {obj['instruction']}
### Input: {obj['input']}
### Response: {obj['output']}"""
                val_data.append({"text": text})

        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        return train_dataset, val_dataset

    def prepare_model(self):
        """Load and prepare model for training"""
        # Determine model source
        if self.model_args.local_model_path and os.path.exists(self.model_args.local_model_path):
            model_path = self.model_args.local_model_path
            print(f"Loading model from local path: {model_path}")
        else:
            model_path = self.model_args.model_name
            print(f"Loading model from Hugging Face: {model_path}")

        print(f"Loading model: {self.model_args.model_name}")

        # Configure model loading for Mac
        model_kwargs = {
            "torch_dtype": torch.float32,
            "device_map": None,  # Disable device_map for Mac
            "trust_remote_code": True,
            "local_files_only": False #bool(self.model_args.local_model_path)
        }

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name,
            trust_remote_code=True,
            local_files_only=False #bool(self.model_args.local_model_path)
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        with mac_config.mutex_guard():
            model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name,
                **model_kwargs
            )

        # Apply LoRA if enabled
        if self.model_args.use_lora:
            print("Applying LoRA configuration...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.model_args.lora_r,
                lora_alpha=self.model_args.lora_alpha,
                lora_dropout=self.model_args.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                inference_mode=False
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # Move model to appropriate device
        model = model.to(self.device)

        return model, tokenizer

    def tokenize_function(self, examples, tokenizer):
        """Tokenize the examples"""
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.data_args.max_length,
            return_tensors="pt"
        )

    @mac_config.handle_data_parallel
    def train(self):
        """Main training function"""
        # Load data
        train_dataset, val_dataset = self.load_data()
        print(f"Loaded {len(train_dataset)} training samples")
        print(f"Loaded {len(val_dataset)} validation samples")

        # Prepare model
        model, tokenizer = self.prepare_model()

        # Tokenize datasets
        with mac_config.mutex_guard():
            train_dataset = train_dataset.map(
                lambda x: self.tokenize_function(x, tokenizer),
                batched=True,
                remove_columns=train_dataset.column_names
            )
            val_dataset = val_dataset.map(
                lambda x: self.tokenize_function(x, tokenizer),
                batched=True,
                remove_columns=val_dataset.column_names
            )

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.training_args.output_dir,
            num_train_epochs=self.training_args.num_train_epochs,
            per_device_train_batch_size=self.training_args.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
            learning_rate=self.training_args.learning_rate,
            warmup_steps=self.training_args.warmup_steps,
            logging_steps=self.training_args.logging_steps,
            eval_steps=self.training_args.eval_steps,
            save_steps=self.training_args.save_steps,
            save_total_limit=self.training_args.save_total_limit,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to=["none"],  # Disable wandb for local training
            fp16=False,  # Disable fp16 on Mac
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,
            label_names=["input_ids"]
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Save model
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.training_args.output_dir)

        print(f"Training complete! Model saved to {self.training_args.output_dir}")

def main():
    # Initialize arguments
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = TrainingArgs()

    # Create trainer
    trainer = RetailAssistantTrainer(model_args, data_args, training_args)

    # Train model
    trainer.train()

if __name__ == "__main__":
    main()