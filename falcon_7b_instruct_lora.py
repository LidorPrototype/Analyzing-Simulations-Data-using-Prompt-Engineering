print("Start...")
print("This is an experiment of doing LoRA finetuning on the Falcon model")
print()

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GenerationConfig)
from trl import SFTTrainer
import time

import warnings
warnings.filterwarnings("ignore")

# login(token=token_key, add_to_git_credential = False, write_permission = True)

# print("-- loging done")

import os
# disable Weights and Biases
os.environ['WANDB_DISABLED']="true"

print("--------------------- imports done ---------------------")
dir_path = 'Tiiuae-falcon-7b-instruct' #TODO
model_name_is = f"peft-dialogue-summary-training___test"
output_dir = f'{dir_path}/{model_name_is}'
model_final_path = f"{output_dir}/final_model/"
training = False

if training:
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
            trust_remote_code=True # It says its better be set to False
    )
    print("--------------------- AutoModelForCausalLM.from_pretrained done ---------------------")

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value"
            # "dense",
            # "dense_h_to_4h",
            # "dense_4h_to_h",
        ],
    )
    print("--------------------- LoraConfig done ---------------------")

    model.config.use_cache = False
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print("--------------------- get_peft_model done ---------------------")

    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("--------------------- AutoTokenizer.from_pretrained done ---------------------")

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim='paged_adamw_32bit',
        save_steps=100,
        fp16=True,
        logging_steps=100,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        max_steps=300,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=True,  # Add this line to disable progress bar
    )
    print("--------------------- TrainingArguments done ---------------------")

    model.config.use_cache = False
    # dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    dataset = load_from_disk("hf_processed_dataset")
    print("--------------------- load_dataset done ---------------------")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
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
    print(f"Training time: {time_taken}")
    print("------------------------------------------------------------")
    print("---------------------- Training Done -----------------------")
    print("------------------------------------------------------------")
    model.save_pretrained(model_final_path) # better option for LangChain loader
    # trainer.save()
    # trainer.push_to_hub()
    print("Model saved.")

else:
    print("\n\n------------------------------------------------------------")
    print("------------------------ Inference -------------------------")
    print("------------------------------------------------------------")

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    print("--------------------- BitsAndBytesConfig done ---------------------")

    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct", quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    print("--------------------- AutoModelForCausalLM.from_pretrained done ---------------------")

    model = PeftModel.from_pretrained(model, model_final_path, local_files_only=True)
    print("--------------------- PeftModel.from_pretrained done ---------------------")

    tok = AutoTokenizer.from_pretrained('tiiuae/falcon-7b-instruct')
    tok.pad_token = tok.eos_token
    print("--------------------- AutoTokenizer.from_pretrained done ---------------------")

    print("--------------------- Input Prompt ---------------------\n")
    def print_in_square(text, max_words):
        words = text.split(' ')
        lines = []
        current_line = []
        for word in words:
            if len(current_line) == max_words or word == '\n':
                lines.append(' '.join(current_line))
                current_line = []
            current_line.append(word)
        if current_line:
            lines.append(' '.join(current_line))
        box_width = max(len(line) for line in lines) + 4
        top_line = '-' * (box_width + 10)
        bottom_line = '-' * (box_width + 10)
        print(top_line)
        for line in lines:
            padded_line = line + ' ' * (box_width - len(line))
            print(f"{'-' * 3}  {padded_line}  {'-' * 3}")
        print(bottom_line)
    

    prompt = """Below is an instruction that describes a task, paired with a context that provides further context. Write a response that appropriately completes the request.

## Instruction:\nOrder the files such that the ones that contribute most to the desired outcome are ranked highest.

## Context:
The data you have is the DataFrame added below, each row represents a different file data in it.
                     filename          duration   amount_sent  average_bandwidth
      flow_info_edge_coloring     14000000000.0 43750000000.0              3.125
               flow_info_ecmp 882415254.2372881 43750000000.0 63.771186440677965
         flow_info_ilp_solver       437500000.0 43750000000.0              100.0
flow_info_simulated_annealing 775820974.5762712 43750000000.0  68.46398305084746
              flow_info_mcvlc       437500000.0 43750000000.0              100.0

## Response:"""
    # print_in_square(prompt, 8)
    print("\n--------------------------------------------------")
    print("\n--------------------- Prompt ---------------------")
    print("\n--------------------------------------------------")
    print(prompt)
    
    print("\n--------------------- Output ---------------------")
    peft_encoding = tok(prompt, return_tensors="pt").to("cuda:0")
    peft_outputs = model.generate(
        input_ids=peft_encoding.input_ids, 
        generation_config=GenerationConfig(
            max_new_tokens=256, 
            pad_token_id = tok.eos_token_id, 
            eos_token_id = tok.eos_token_id, 
            attention_mask = peft_encoding.attention_mask, 
            temperature=0.1, 
            top_p=0.1, 
            repetition_penalty=1.2, 
            num_return_sequences=1,
        )
    )
    peft_text_output = tok.decode(peft_outputs[0], skip_special_tokens=True)
    # peft_text_output = peft_text_output.replace('\n', '  ')
    print("\n--------------------------------------------------")
    print("\n-------------------- Response --------------------")
    print("\n--------------------------------------------------")
    print(peft_text_output)
    # print_in_square(peft_text_output, 8)

print()
print("------------------------------------------------------------")
print("-------------------------- Done! ---------------------------")
print("------------------------------------------------------------")
