import os
import random
import pandas as pd
from datasets import Dataset, load_from_disk

def format_in_template(in_data: str, out_data: str):
    user_input_questions = [
        "Prioritize the files based on their overall efficiency.",
        "Rank the files considering a balance between speed and resource usage.",
        "Order the files such that the most optimal ones appear first.",
        "Identify the files that are most suitable for processing given the current constraints.",
        "Considering all available data, rank the files from best to worst.",
        "Order the files such that the ones with the most desirable qualities are at the top.",
        "Rank the files prioritizing those that would be most beneficial to process.",
        "Identify the files that stand out from the others based on the provided information.",
        "Considering all relevant factors, rank the files for optimal performance.",
        "Order the files in a way that maximizes the benefit of processing them.",
        "Rank the files such that the most effective ones are prioritized.",
        "Considering the available data, order the files for optimal selection.",
        "Identify the files that are most likely to yield the best results.",
        "Rank the files such that the ones with the most potential are at the top.",
        "Order the files for processing in a way that optimizes resource allocation.",
        "Rank the files considering their overall suitability for the task at hand.",
        "Considering all factors, prioritize the files that are most likely to be successful.",
        "Order the files such that the ones that contribute most to the desired outcome are ranked highest.",
        "Rank the files based on the information provided, prioritizing those that are most valuable.",
        "Considering all relevant aspects, order the files for optimal processing efficiency.",
    ]
    ZeroShotInstruct = """Below is an instruction that describes a task, paired with a context that provides further context. Write a response that appropriately completes the request.

## Instruction:\n{user_input}

## Context:
The data you have is the DataFrame added below, each row represents a different file data in it.
{data}

## Response:\n{output}"""
    ZeroShotGeneral = """You are a researcher assistant, answer truthfully about the data provided to you below.
Answer the question based on the context below. If the question cannot be answered using the information provided answer with "I don't know".

## Context:
The data provided below is a DataFrame summary of 5 different CSV files\n
{data}

## User: \n{user_input}

## Assistant:\n{output}"""
    RawPrompt = """## Data:
The data provided below is a DataFrame summary of 5 different CSV files\n
{data}

## User: {user_input}

## Assistant:\n{output}"""
    prompts = [ZeroShotInstruct, ZeroShotGeneral, RawPrompt]
    return random.choice(prompts).format(user_input=random.choice(user_input_questions), data=in_data, output=out_data)
    # return PROMPT_2.format(user_input=random.choice(user_input_questions), data=in_data), out_data # This without the output key

def process_csv(csv_path):
    with open(csv_path, "r") as csvfile:
        lines = csvfile.readlines()
        data = lines[1:]
        data = [row.strip() for row in data]
    df = pd.DataFrame([row.split(",") for row in data], columns=lines[0].replace("\n", "").split(","))
    input_data = df[["filename", "duration", "amount_sent", "average_bandwidth"]]
    input_data = input_data.to_string(index=False)
    output_data = df[["filename", "order"]]
    output_data = output_data.to_string(index=False)
    return format_in_template(in_data=input_data, out_data=output_data)

def save_processed_dataset(dataset_to_save, path: str = "hf_processed_dataset"):
    dataset_to_save.save_to_disk(path)

def load_processed_dataset(path: str = "hf_processed_dataset") -> Dataset:
    loaded_dataset = load_from_disk(path) # load_dataset(path)
    return loaded_dataset

print("Start...\n")
dataset_path = "/home/lidorel/lora-tests/dataset_organized"
processed_dataset_path = "/home/lidorel/lora-tests/hf_processed_dataset"
data = []
i = 1
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    csv_path = os.path.join(folder_path, "files_rank.csv")
    # input_data, output_data = process_csv(csv_path)
    # data.append({"input": input_data, "output": output_data})
    text = process_csv(csv_path)
    data.append({"text": text})
    print(f"{i}) Processed input files from folder: {folder_path}")
    i+=1
dataset = Dataset.from_list(data)
save_processed_dataset(dataset_to_save=dataset, path=processed_dataset_path)
print("*********************************************************************************")
print("***************************** Loading Saved Dataset *****************************")
print("*********************************************************************************")
dataset = load_processed_dataset(path=processed_dataset_path)
j = 0
for example in dataset:
    if j == 20: break
    print(f"Text: \n{example['text']}")
    # print(f"Input: \n{example['input']}")
    # print(f"Output: \n{example['output']}")
    print("-" * 120)
    j += 1

