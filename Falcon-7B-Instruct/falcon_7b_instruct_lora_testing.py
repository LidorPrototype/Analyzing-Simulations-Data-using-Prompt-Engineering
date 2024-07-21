print("\n*************************************************************************************")
print("***************************** Testing Code Starting *********************************")
print("*************************************************************************************")

import torch
import time
import difflib
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
logs_dir = f'{dir_path}/logs'
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
    hf_model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=False
)
print("--------------------- AutoModelForCausalLM.from_pretrained done ---------------------")

model = PeftModel.from_pretrained(model, model_final_path, local_files_only=True)
print("--------------------- PeftModel.from_pretrained done ---------------------")

tok = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=False)
tok.pad_token = tok.eos_token
print("--------------------- AutoTokenizer.from_pretrained done ---------------------")

# dataset_name = "hf_test_dataset_top5"
dataset_name = "hf_test_dataset_top1"
dataset = load_from_disk(dataset_name)
print("\n\n-------------------------------------------------------------")
print("-------------------------- Testing --------------------------")
print("-------------------------------------------------------------")

def split_prompt_text(text: str, push: int = 1, get_last: bool = False):
    """
    Splits the full text and the response dataframe
    Args:
      text: The complete response string of the prompt and response.
    Returns:
      tuple: The prompt for the model and the desired response dataframe
    if `get_last` was set to True, it will return all the text after the final ':' with strip()
    """
    if get_last:
        last_colon_index = text.rfind(':')
        if last_colon_index != -1:
            return text[: last_colon_index], text[last_colon_index + 1:].strip()
        else:
            return ""
    lines = text.splitlines()
    # Identify potential data sections based on structure
    data_sections = []
    for i, line in enumerate(lines):
        if any(char.isupper() for char in line.strip()) and not line.startswith(" "):
            data_sections.append(i)
    if not data_sections:
        return None  # No data sections found
    last_dataframe_index = data_sections[-1]
    dataframe_text = "\n".join(lines[last_dataframe_index + 1 + push:])
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

def evaluate_example(true_df, predicted_df):
  correct_predictions_amount = 0
  correct_predictions = []
  for i in range(len(true_df)):
    rank_diff = abs(true_df.iloc[i]["order"] - predicted_df.iloc[i]["order"])
    if rank_diff == 0:
      correct_predictions_amount += 1
      correct_predictions.append(true_df.iloc[i]["order"])
  incorrect_predictions_amount = len(true_df) - correct_predictions_amount
  return correct_predictions_amount, incorrect_predictions_amount, correct_predictions

def levenshtein_distance(true_string, predicted_string):
    """
    This function compares the similarity between two strings using the Levenshtein distance.
    Args:
        true_string: The true or reference string.
        predicted_string: The predicted string from the LLM.
    Returns:
        The Levenshtein distance between the strings, representing the number of edits needed to transform one into the other.
    """
    # Use difflib.SequenceMatcher to calculate Levenshtein distance
    matcher = difflib.SequenceMatcher(None, true_string, predicted_string)
    return matcher.quick_ratio() # TODO Make it binary return

def top_5_testing(_dataset):
    kendall = []
    spearman = []
    err, succ, skip = 0, 0, 0
    for item in tqdm(_dataset, desc="Testing model", disable=True):
        try:
            full_text = item["text"]
            i = 0
            while i < 4:
                prompt, true_df = split_prompt_text(full_text, push=i)
                i += 1
                if (true_df.columns.to_list()) == ['filename', 'order']: break
            if (true_df.columns.to_list()) != ['filename', 'order']:
                skip += 1
                continue
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
            kendall_tau, p_value, spearman_rho, p_value = 0, 0, 0, 0
            i = 1
            while i < 4:
                try:
                    predicted_prompt, predicted_df = split_prompt_text(peft_text_output, push = i)
                    tmp = predicted_df["order"]
                    break
                except:
                    i += 1
            if (predicted_df.columns.to_list()) != ['filename', 'order']: 
                skip += 1
                continue
            _ = true_df["order"]
            _ = predicted_df["order"]
            kendall_tau, p_value = kendalltau(true_df["order"], predicted_df["order"])
            spearman_rho, p_value = spearmanr(true_df["order"], predicted_df["order"])
            kendall.append(kendall_tau)
            spearman.append(spearman_rho)
            succ += 1
            correct_preds, incorrect_preds, correct_ranks = evaluate_example(true_df, predicted_df)
            print(f"Correct Predictions: {correct_preds}/5")
            print(f"Incorrect Predictions: {incorrect_preds}/5")
            print(f"Correct Ranks: {correct_ranks}")
            print("**************************************************")
        except Exception as e:
            print("******************* ERROR ABOVE *******************")
            err += 1
    print("\n\n\n**************************************************")
    print("**************************************************")
    print(f"kendall avg: {sum(kendall) / len(kendall) :.2}")
    print(f"spearman avg: {sum(spearman) / len(spearman) :.2f}")
    print(f"{succ = }")
    print(f"{err = }")
    print(f"{skip = }")
    print("**************************************************\n")

def top_1_testing(_dataset):
    scores = []
    orders = []
    err, succ, skip = 0, 0, 0
    all_predictions = {}
    for item in tqdm(_dataset, desc="Testing model", disable=True):
        try:
            full_text = item["text"]
            i = 0
            while i < 4:
                prompt, true_file = split_prompt_text(full_text, get_last=True)
                true_file = str(true_file)
                i += 1
                if '\n' not in true_file: break
            if '\n' in true_file:
                skip += 1
                continue
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
            i = 1
            while i < 4:
                try:
                    predicted_prompt, predicted_file = split_prompt_text(peft_text_output, get_last=True)
                    predicted_file = str(predicted_file)
                    if '\n' not in true_file: break
                except:
                    i += 1
            if '\n' in predicted_file: 
                skip += 1
                continue
            score = levenshtein_distance(true_string=true_file.strip(), predicted_string=predicted_file.strip())
            succ += 1
            correct_order = true_file.strip() == predicted_file.strip()
            print(f"Levenshtein Distance: {score}/1.0")
            print(f"Correct Characters Order: {correct_order}")
            print("**************************************************")
            scores.append(score)
            orders.append(correct_order)
            t = true_file.strip()
            p = predicted_file.strip()
            if t not in all_predictions:
                all_predictions[t] = []
            all_predictions[t].append(p)
        except Exception as e:
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            print("******************* ERROR ABOVE *******************")
            err += 1
            break
    print("\n\n\n**************************************************")
    print("**************************************************")
    print(f"score avg: {sum(scores) / len(scores) :.2}")
    print(f"Number of True values: {orders.count(True)}")
    print(f"Number of False values: {orders.count(False)}")
    print(f"{succ = }")
    print(f"{err = }")
    print(f"{skip = }")
    print("**************************************************\n")
    data_all = {
        "scores": scores,
        "orders": orders,
        "all_predictions": all_predictions
    }
    import json
    with open("data_all.json", "wt") as f:
        # Dump the dictionary to the file as JSON
        json.dump(data_all, f)

start = time.time()

# top_5_testing(dataset)
top_1_testing(dataset)

end=time.time()
time_taken=end-start
seconds = int(time_taken)
hours = seconds // 3600
minutes = (seconds % 3600) // 60
seconds = seconds % 60
print("------------------------------------------------------------")
print("---------------------- Testing Done ------------------------")
print(f"----------------- Testing time: {hours:02d}:{minutes:02d}:{seconds:03d} ----------------")
print("------------------------------------------------------------")

print("\n\n")
print("------------------------------------------------------------")
print("-------------------------- Done! ---------------------------")
print("------------------------------------------------------------")
