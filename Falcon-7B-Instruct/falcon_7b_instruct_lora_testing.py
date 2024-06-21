print("*************************************************************************************")
print("*************************************************************************************")
print("***************************** Testing Code Starting *********************************")
print("*************************************************************************************")
print("*************************************************************************************")

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig)
from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd
from io import StringIO
from scipy.stats import kendalltau
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['WANDB_DISABLED']="true"

print("--------------------- imports done ---------------------")
hf_model_name = "tiiuae/falcon-7b-instruct"
dir_path = 'Tiiuae-falcon-7b-instruct'
model_name_is = f"peft-training"
output_dir = f'{dir_path}/{model_name_is}'
model_final_path = f"{output_dir}/final_model/"

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
    hf_model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
print("--------------------- AutoModelForCausalLM.from_pretrained done ---------------------")

model = PeftModel.from_pretrained(model, model_final_path, local_files_only=True)
print("--------------------- PeftModel.from_pretrained done ---------------------")

tok = AutoTokenizer.from_pretrained(hf_model_name)
tok.pad_token = tok.eos_token
print("--------------------- AutoTokenizer.from_pretrained done ---------------------")

dataset = load_from_disk("hf_test_dataset")
print("\n\n-------------------------------------------------------------")
print("-------------------------- Testing --------------------------")
print("-------------------------------------------------------------")

def split_prompt_text(text, full: bool = False):
    """
    Splits the full text and the response dataframe
    Args:
      text: The complete response string of the prompt and response.
    Returns:
      tuple: The prompt for the model and the desired response dataframe
    """
    lines = text.splitlines()
    # Identify potential data sections based on structure
    data_sections = []
    for i, line in enumerate(lines):
        if any(char.isupper() for char in line.strip()) and not line.startswith(" "):
            data_sections.append(i)
    if not data_sections:
        return None  # No data sections found
    last_dataframe_index = data_sections[-1]
    if full:
        dataframe_text = "\n".join(lines[last_dataframe_index + 3:])
    else:
        dataframe_text = "\n".join(lines[last_dataframe_index + 1:])
    try:
        dataframe = pd.read_csv(StringIO(dataframe_text), delim_whitespace=True)
    except ValueError:
        pass
    cols = []
    for col in dataframe.columns:
        title = col.strip().lower()
        for t in title.split(' '):
            cols.append(t)
    dataframe.columns = cols
    return text[:-len(dataframe_text)], dataframe

kendall = []
spearman = []
err = 0
for item in tqdm(dataset, desc="Testing model", disable=True):
    try:
        full_text = item["text"]
        prompt, true_df = split_prompt_text(full_text, full=True)
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
        predicted_prompt, predicted_df = split_prompt_text(peft_text_output)
        kendall_tau, p_value = kendalltau(true_df["order"], predicted_df["order"])
        spearman_rho, p_value = spearmanr(true_df["order"], predicted_df["order"])
        kendall.append(kendall_tau)
        spearman.append(spearman_rho)
    except: err += 1
print("\n**************************************************")
print(f"kendall avg: {sum(kendall) / len(kendall)}")
print(f"spearman avg: {sum(spearman) / len(spearman)}")
print(f"err: {err}")
print("**************************************************\n")

print("\n\n")
print("------------------------------------------------------------")
print("-------------------------- Done! ---------------------------")
print("------------------------------------------------------------")
