import os
import pandas as pd
from datasets import Dataset, load_from_disk

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
    return input_data, output_data

def save_processed_dataset(dataset_to_save, path: str = "hf_processed_dataset"):
    dataset_to_save.save_to_disk(path)

def load_processed_dataset(path: str = "hf_processed_dataset") -> Dataset:
    loaded_dataset = load_from_disk(path)
    # loaded_dataset = load_dataset(path)
    return loaded_dataset

print("Start...\n")
dataset_path = "/home/lidorel/lora-tests/dataset_organized"
processed_dataset_path = "/home/lidorel/lora-tests/hf_processed_dataset"
data = []
i = 1
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    csv_path = os.path.join(folder_path, "files_rank.csv")
    input_data, output_data = process_csv(csv_path)
    data.append({"input": input_data, "output": output_data})
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
    if j == 10: break
    print(f"Input: {example['input']}")
    print(f"Output: {example['output']}")
    print("-" * 30)
    j += 1
