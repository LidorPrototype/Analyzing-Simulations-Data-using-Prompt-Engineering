print("*************************************************************************************")
print("*************************************************************************************")
print("**************************** Training Code Starting *********************************")
print("*************************************************************************************")
print("*************************************************************************************")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, TrainingArguments
from datasets import load_from_disk
from trl import SFTTrainer
from peft import LoraConfig
import os, time
import warnings
# from accelerate import Accelerator # TESTING

os.environ['WANDB_DISABLED'] = "true" # disable Weights and Biases
warnings.filterwarnings("ignore")

# accelerator = Accelerator() # TESTING
TTT = "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"
model_id = "google/gemma-2b-it"
dir_path = 'Google-gemma-2b-it'
model_name_is = f"peft-training"
output_dir = f'{dir_path}/{model_name_is}'
EPOCHS = 4000
LOGS = 1000
SAVES = 1000

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
print("--------------------- BitsAndBytesConfig done ---------------------")

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=TTT, trust_remote_code=False)
print("--------------------- AutoModelForCausalLM.from_pretrained done ---------------------")

dataset = load_from_disk("hf_processed_dataset")
train_size = int(0.8 * len(dataset))
test_dataset = dataset.select(range(train_size, len(dataset)))
train_dataset = dataset.select(range(train_size))
print("--------------------- load datasets done --------------------")

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=['q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'k_proj', 'v_proj'],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
print("--------------------- LoraConfig done ---------------------")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=TTT, trust_remote_code=False)
# tokenizer.pad_token = tokenizer.eos_token
print("--------------------- AutoTokenizer.from_pretrained done ---------------------")

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit", # paged_adamw_32bit
    learning_rate=0.0002, # 0.001
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    logging_steps=LOGS,
    save_steps=SAVES,
    max_steps=EPOCHS,
    disable_tqdm=True,
    # fsdp="full_shard",
    # fsdp_config="fsdp_config.json"
)
print("--------------------- TrainingArguments done ---------------------")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_arguments,
    peft_config=lora_config,
    max_seq_length = 512,
    tokenizer=tokenizer,
    dataset_text_field="text",
)
print("--------------------- SFTTrainer done ---------------------")

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

prompt = """You are a researcher assistant, answer truthfully about the data provided to you below.
Answer the question based on the context below. If the question cannot be answered using the information provided answer with "I don't know".

## Context:
The data provided below is a DataFrame summary of 5 different CSV files

                     filename     duration  amount_sent average_bandwidth
      flow_info_edge_coloring 2800000000.0 8750000000.0             3.125
         flow_info_ilp_solver   87500000.0 8750000000.0             100.0
               flow_info_ecmp  172550000.0 8750000000.0             66.12
              flow_info_mcvlc   87500000.0 8750000000.0             100.0
flow_info_simulated_annealing  161525000.0 8750000000.0             66.89

## User: 
Identify the files that are most suitable for processing given the current constraints.

## Assistant:"""

print("\n--------------------------------------------------")
print("\n--------------------- Prompt ---------------------")
print("\n--------------------------------------------------")
print(prompt)

peft_encoding = tokenizer(prompt, return_tensors="pt").to("cuda:0")
peft_outputs = model.generate(
    input_ids=peft_encoding.input_ids, 
    generation_config=GenerationConfig(
        max_new_tokens=256, 
        pad_token_id = tokenizer.eos_token_id, 
        eos_token_id = tokenizer.eos_token_id, 
        attention_mask = peft_encoding.attention_mask, 
        temperature=0.1, 
        top_p=0.1, 
        repetition_penalty=1.2, 
        num_return_sequences=1,
    )
)
peft_text_output = tokenizer.decode(peft_outputs[0], skip_special_tokens=True)
print("\n--------------------------------------------------")
print("\n-------------------- Response --------------------")
print("\n--------------------------------------------------")
print(peft_text_output)

print()
print("------------------------------------------------------------")
print("-------------------------- Done! ---------------------------")
print("------------------------------------------------------------")