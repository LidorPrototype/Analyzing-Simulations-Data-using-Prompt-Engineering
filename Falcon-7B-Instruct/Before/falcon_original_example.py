from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

def print_in_square(text, max_words = 8, need_split: bool = True):
    if need_split:
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
    else:
        lines = text.splitlines()
        box_width = max(len(line) for line in lines) + 4
        top_line = '-' * (box_width + 10)
        bottom_line = '-' * (box_width + 10)
        print(top_line)
        for line in lines:
            padded_line = line + ' ' * (box_width - len(line))
            print(f"{'-' * 3}  {padded_line}  {'-' * 3}")
        print(bottom_line)

def load_and_rank_data(filepaths, weights=None):
    dataframes = []
    for filepath in filepaths:
        try:
            df = pd.read_csv(filepath)
            summary_df = df.mean(axis=0).to_frame().T
            dataframes.append(summary_df)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
    if not dataframes:
        print("No valid CSV files found. Returning empty DataFrame.")
        return pd.DataFrame()
    consolidated_df = pd.concat(dataframes, ignore_index=True)
    consolidated_df.drop(columns=['start_time'], inplace=True)
    if weights:
        consolidated_df['score'] = (consolidated_df * weights).mean(axis=1)
    else:
        consolidated_df['score'] = consolidated_df.mean(axis=1)
    ranked_df = consolidated_df.sort_values(by='score', ascending=True)
    ranked_df = ranked_df.reset_index(drop=True)
    ranked_df['rank'] = ranked_df.shape[0] - ranked_df.reset_index()['index']
    rank_dict = {}
    for index, row in ranked_df.iterrows():
        rank_dict[filepaths[index].split('/')[-1]] = row['rank']
    ranked_df = ranked_df.sort_values(by='rank').reset_index(drop=True)
    return ranked_df, rank_dict


filenames = [
    "/home/lidorel/lora-tests/dataset_organized/s_0_cj_1_0_cf_rs_2_BLOOM_P1/flow_info_ecmp.csv",
    "/home/lidorel/lora-tests/dataset_organized/s_0_cj_1_0_cf_rs_2_BLOOM_P1/flow_info_simulated_annealing.csv",
    "/home/lidorel/lora-tests/dataset_organized/s_0_cj_1_0_cf_rs_2_BLOOM_P1/flow_info_edge_coloring.csv",
    "/home/lidorel/lora-tests/dataset_organized/s_0_cj_1_0_cf_rs_2_BLOOM_P1/flow_info_ilp_solver.csv",
    "/home/lidorel/lora-tests/dataset_organized/s_0_cj_1_0_cf_rs_2_BLOOM_P1/flow_info_mcvlc.csv"
]
data = load_and_rank_data(filenames)
prompt = "The following DataFrame shows average data from 5 simulations. (each row is a different file)"
prompt += "Analyze the data to determine which file suggests the most efficient and reliable protocol based on the simulation results. Explain your reasoning considering all relevant factors from the data provided.\n"
prompt += f"\n{data[0]}\n"

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
attention_mask = torch.ones_like(input_ids)
gen_tokens = model.generate(input_ids, max_length=3000, attention_mask=attention_mask, return_dict_in_generate=True)['sequences']
decoded_text = tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

print_in_square(prompt, need_split=False)
print()
print_in_square(decoded_text.replace('\n', '  '), need_split=True, max_words=15)
