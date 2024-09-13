#### Getting started

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import gc

#### Loading dataset, model, and tokenizer
data_dir = '/gpfs/home/dcolombaro/Jailbreak_LLM-main/data'

model_dir = 'swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA'

save_dir = '/gpfs/home/dcolombaro/Jailbreak_LLM-main/models/Llamantino3-8b'

dataset = load_dataset("json", data_files= data_dir + "/data_new_ita.jsonl", split="train")

token = '' # Add your HuggingFace token

#### 8-bit quantization configuration

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

#### Loading Llama-2 model

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=quant_config,
    device_map="auto",
    ignore_mismatched_sizes=True,
    token=token)

model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.pad_token_id = model.config.eos_token_id
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

#### Loading tokenizer

tokenizer = AutoTokenizer.from_pretrained("swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA", token = token, truncation=True, do_lower_case=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#### PEFT parameters

peft_params = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=64, #32
    bias="none",
    task_type="CAUSAL_LM",
    target_modules = ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

peft_model = get_peft_model(model, peft_params)
#### Training parameters

training_params = TrainingArguments(
    output_dir=save_dir,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.1, #0.03
    group_by_length=True,
    lr_scheduler_type="linear",
)

#### Model fine-tuning

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

check_dir = save_dir + '/checkpoint-1150'

model.save_pretrained(save_dir, save_adapter=True, save_config=True)

model = AutoModelForCausalLM.from_pretrained(
    "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA",
    quantization_config=quant_config,
    device_map="auto",
    ignore_mismatched_sizes=True,
    token=token)

# model.resize_token_embeddings(len(tokenizer))

device_map = infer_auto_device_map(model)
print(device_map)

merge_model = PeftModel.from_pretrained(model, check_dir)
merged_model = merge_model.merge_and_unload()
merged_model.save_pretrained(save_dir + '/merged')
tokenizer.save_pretrained(save_dir + '/merged')