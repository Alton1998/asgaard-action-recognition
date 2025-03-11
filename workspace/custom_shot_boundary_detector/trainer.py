import json
from transformers import LlavaProcessor, LlavaForConditionalGeneration, TrainingArguments
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from trl import SFTTrainer

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model and processor
model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, load_in_4bit=True)
model.to(device)


def collate_fn(batch):
    """Processes a batch without moving tensors to CUDA (DataLoader runs on CPU)."""
    batch_texts = []
    batch_images = []

    for entry in batch:
        batch_texts.append(entry["text"])
        for image in entry["frames"]:
            batch_images.append(Image.open(image))

    # Process batch using the processor (KEEP IT ON CPU)
    processed_data = processor(
        text=batch_texts,
        images=batch_images,
        return_tensors="pt",
        padding=True,
    )

    labels = processed_data["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask image tokens
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    processed_data["labels"] = labels

    return processed_data  # Keep everything on CPU for DataLoader


def add_text(data):
    """Adds a 'text' key to each data sample."""
    data["text"] = data["prompt"]
    return data


if __name__ == "__main__":
    # Load and preprocess dataset
    dataset = load_dataset("json", data_files="processed_data.json", split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(add_text)
    train_test_split = dataset.train_test_split(test_size=0.2)

    # Define LoRA configuration
    target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2"]
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )

    # Apply PEFT
    peft_model = get_peft_model(model, peft_config).to(device)
    print("Trainable Parameters are as follows:")
    peft_model.print_trainable_parameters()

    # Set up training arguments
    print("Setting up training Arguments")
    training_args = TrainingArguments(
        output_dir='./asgaard-model-finetuning-result-alton',
        num_train_epochs=4,
        gradient_accumulation_steps=32,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=10,
        weight_decay=0.01,
        evaluation_strategy='steps',
        eval_steps=10,
        logging_steps=1,
        logging_strategy="steps",
        gradient_checkpointing=True,
        save_steps=500,
        remove_unused_columns=False  # Important to avoid key mismatches
    )

    # Initialize Trainer (SFTTrainer will handle DataLoader internally)
    trainer = SFTTrainer(
        model=peft_model,  # Use LoRA-wrapped model
        args=training_args,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
        data_collator=collate_fn,  # Pass collate function
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    # Start Training
    print("Starting Training")
    trainer.train()
