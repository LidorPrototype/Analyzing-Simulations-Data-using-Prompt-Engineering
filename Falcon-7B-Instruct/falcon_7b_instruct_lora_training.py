
print("*************************************************************************************")
print("*************************************************************************************")
print("**************************** Training Code Starting *********************************")
print("*************************************************************************************")
print("*************************************************************************************")

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer
import time
import os
import warnings

# login(token=token_key, add_to_git_credential = False, write_permission = True)
# print("-- loging done")

os.environ['WANDB_DISABLED'] = "true" # disable Weights and Biases
warnings.filterwarnings("ignore")
print("--------------------- Imports & Settings done ---------------------")

hf_model_name = "tiiuae/falcon-7b-instruct"
dir_path = 'Tiiuae-falcon-7b-instruct'
model_name_is = f"peft-training"
output_dir = f'{dir_path}/{model_name_is}'
model_final_path = f"{output_dir}/final_model/"
EPOCHS = 3500
LOGS = 700
SAVES = 700

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
print("--------------------- BitsAndBytesConfig done ---------------------")

model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct",
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=False
)
print("--------------------- AutoModelForCausalLM.from_pretrained done ---------------------")

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05, # 0.1
    r=64,
    bias="lora_only", # none
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value"
    ],
)
print("--------------------- LoraConfig done ---------------------")

model.config.use_cache = False
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print("--------------------- get_peft_model done ---------------------")

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=False)
tokenizer.pad_token = tokenizer.eos_token
print("--------------------- AutoTokenizer.from_pretrained done ---------------------")

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim='paged_adamw_32bit',
    save_steps=SAVES,
    fp16=True,
    logging_steps=LOGS,
    learning_rate=0.001,
    max_grad_norm=0.3,
    max_steps=EPOCHS,
    warmup_ratio=0.15, # 0.03
    lr_scheduler_type="constant",
    disable_tqdm=True,
)
print("--------------------- TrainingArguments done ---------------------")

model.config.use_cache = False
# dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
dataset = load_from_disk("hf_processed_dataset")
train_size = int(0.8 * len(dataset))
test_dataset = dataset.select(range(train_size, len(dataset)))
train_dataset = dataset.select(range(train_size))
test_dataset.save_to_disk("hf_test_dataset")
print("--------------------- load datasets done --------------------")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=True,
)
print("--------------------- SFTTrainer done ---------------------")

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

print("------------------------------------------------------------")
print("--------------------- Training Started ---------------------")
print("------------------------------------------------------------")
start = time.time()
trainer.train()
end=time.time()
time_taken=end-start
seconds = int(time_taken)
hours = seconds // 3600
minutes = (seconds % 3600) // 60
seconds = seconds % 60
print("------------------------------------------------------------")
print("---------------------- Training Done -----------------------")
print(f"---------------- Training time: {hours:02d}:{minutes:02d}:{seconds:03d} ----------------")
print("------------------------------------------------------------")


print("------------------------------------------------------------")
print("--------------------- Evaluating Model ---------------------")
print("------------------------------------------------------------")
results = trainer.evaluate()
print(results)
print()


model.save_pretrained(model_final_path) # better option for LangChain loader
# trainer.save()
# trainer.push_to_hub()
print("Model saved.")

print()
print("------------------------------------------------------------")
print("-------------------------- Done! ---------------------------")
print("------------------------------------------------------------")
