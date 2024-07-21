import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig)

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['WANDB_DISABLED']="true"

print("--------------------- imports done ---------------------")
hf_model_name = "tiiuae/falcon-7b-instruct"
dir_path = 'Tiiuae-falcon-7b-instruct'
model_name_is = f"peft-training"
output_dir = f'{dir_path}/{model_name_is}'
logs_dir = f'{dir_path}/logs'
model_final_path = f"{output_dir}/final_model/"
after_merge_model_path = "Tiiuae-falcon-7b-instruct-final/"

print("\n\n------------------------------------------------------------")
print("---------------------- Configurations ----------------------")
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
    hf_model_name, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=False
)
print("--------------------- AutoModelForCausalLM.from_pretrained done ---------------------")

lora_model = PeftModel.from_pretrained(model, model_final_path, local_files_only=True)
print("--------------------- PeftModel.from_pretrained done ---------------------")

tok = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=False)
tok.pad_token = tok.eos_token
print("--------------------- AutoTokenizer.from_pretrained done ---------------------")


print("--------------------- lora_model.merge_and_unload() start --------------------")
lora_model.merge_and_unload()
print("--------------------- lora_model.merge_and_unload() done ---------------------")

# Save the combined model locally
lora_model.save_pretrained(after_merge_model_path, from_pt=True)
print("Model saved.")
tok.save_pretrained(after_merge_model_path, from_pt=True)
print("Tokenizer saved.")

print("--------------------- Model Loading ---------------------")
local_model = AutoModelForCausalLM.from_pretrained(after_merge_model_path)
print("Model state_dict loaded.")
print("------------------- Tokenizer Loading -------------------")
local_tokenizer = AutoTokenizer.from_pretrained(after_merge_model_path)
print("Tokenizer loaded.")
print("---------------------- Loading Done ---------------------")



print("\n\n------------------------------------------------------------")
print("----------------------- Generating -------------------------")
print("------------------------------------------------------------")

prompt = """The following is a task to find the optimal algorithm for the flow completion time metric.

Consider the following DataFrame where each row is a different algorithm,

DataFrame:
                     filename     duration        amount_sent average_bandwidth
               flow_info_ecmp  35290020.85       2187500001.8 75.87544370616165
         flow_info_ilp_solver   21875000.0       2187500000.0             100.0
      flow_info_edge_coloring  700000000.0       2187500000.0             3.125
flow_info_simulated_annealing 35177734.214 2187500000.4333334 75.47932859939641
              flow_info_mcvlc   23275000.0       2187500000.0              96.8

Instruction:
Rank the most effective file.

Answer:
"""

peft_encoding = local_tokenizer(prompt, return_tensors="pt").to("cuda:0")
peft_outputs = local_model.generate(
    input_ids=peft_encoding.input_ids, 
    generation_config=GenerationConfig(
        max_new_tokens=256, 
        pad_token_id = local_tokenizer.eos_token_id, 
        eos_token_id = local_tokenizer.eos_token_id, 
        attention_mask = peft_encoding.attention_mask, 
        temperature=0.1, 
        top_p=0.1, 
        repetition_penalty=1.2, 
        num_return_sequences=1,
    )
)
peft_text_output = local_tokenizer.decode(peft_outputs[0], skip_special_tokens=True)

print("\n\n------------------------------------------------------------")
print("------------------------- Output ---------------------------")
print("------------------------------------------------------------")

print()
print(peft_text_output)

print("\n\n------------------------------------------------------------")
print("----------------------- True Value -------------------------")
print("------------------------------------------------------------")
print("Solution should be: <flow_info_ilp_solver>")