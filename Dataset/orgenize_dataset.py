import os, shutil
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

def organize_dataset(source_path, destination_path):
  """
  Organizes a complex dataset with nested folders into a structure with one folder per input combination, 
    with shorter and informative names.
  Args:
    source_path: Path to the original dataset.
    destination_path: Path to the new organized dataset.
  """
  num_csv_files = 0
  num_folders = 0
  for root, dirs, files in os.walk(source_path):
    # Skip the source directory itself
    if root == source_path:
      continue
    if len(files) < 4:
      continue
    # Extract input parameters and shorten names
    params = []
    for part in root.split(os.sep)[1:]:
      short_name = {
          "seed": "s",
          "concurrent_jobs": "cj",
          "core_failures": "cf",
          "ring_size": "rs",
          "different_ring_sizes": "drz"
      }.get(part.lower(), part)  # Use short names if available
      params.append(short_name)
    # Construct the new folder name (shorter with underscores)
    new_folder_name = "_".join(params)
    new_folder_name = new_folder_name.replace("seed", "s")\
                                        .replace("concurrent_jobs", "cj")\
                                        .replace("core_failures", "cf")\
                                        .replace("ring_size", "rs")\
                                        .replace("different_ring_sizes", "drs")\
                                        .replace("organized_mid_", "")\
                                        .replace("dataset_organized_mid_", "")\
                                        .replace("home_lidorel_lora-tests_dataset_", "")
    
    # Create the destination folder only if there are CSV files
    has_csv = any(filename.endswith(".csv") for filename in files)
    if has_csv:
      dest_folder = os.path.join(destination_path, new_folder_name)
      os.makedirs(dest_folder, exist_ok=True)
      num_folders += 1
      # Copy CSV files to the destination folder
      for filename in files:
        if filename.endswith(".csv"):
          source_file = os.path.join(root, filename)
          dest_file = os.path.join(dest_folder, filename)
          shutil.copy2(source_file, dest_file)
          num_csv_files += 1
  print(f"Dataset organized successfully! Check {destination_path}")
  print(f"Total CSV files: {num_csv_files}")
  print(f"Total folders created: {num_folders}")

def add_headers_to_files(dataset_folder, header_file, desired_columns):
    # Read the header file
    with open(header_file, 'r') as f:
        headers = f.readline().strip().split(',')
    # Iterate over each folder in the dataset folder
    i = 1
    for folder in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, folder)
        # Check if the item in the dataset folder is a directory
        if os.path.isdir(folder_path):
            # Iterate over each file in the folder
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path, header=None)
                # Assign headers to the DataFrame columns
                df.columns = headers[:len(df.columns)]
                # Filter the DataFrame to include only desired columns
                df = df[desired_columns]
                # Write the DataFrame back to the CSV file
                df.to_csv(file_path, index=False)
                print(f"{i}) Added headers to and saved desired columns in {file_path}")
                i += 1

def split_data(dataset_path, chunk_size=1000):
    jk = 1
    counter = 1
    # Iterate through each folder in the dataset
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            # Iterate through each CSV file in the folder
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith(".csv"):
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    total_rows = len(df)
                    # Calculate number of chunks
                    num_chunks = (total_rows + chunk_size - 1) // chunk_size
                    # Create t folders if they don't exist
                    for i in range(1, num_chunks + 1):
                        chunk_folder = os.path.join(folder_path, f"P{i}")
                        os.makedirs(chunk_folder, exist_ok=True)
                    # Split data into chunks
                    j = 0
                    for j in range(num_chunks):
                        start_row = j * chunk_size
                        end_row = min((j + 1) * chunk_size, total_rows)
                        chunk_df = df.iloc[start_row:end_row]
                        # Write chunk data to new CSV file
                        chunk_file_path = os.path.join(folder_path, f"P{j+1}", file)
                        chunk_df.to_csv(chunk_file_path, index=False)
                        counter += 1
                    print(f"{jk}) Got splitted into {j} parts.")
                    jk += 1
                    # Delete original CSV file
                    os.remove(file_path)
    print(f"Number of files int otal are: {counter}")

def rank_dataset(dataset_folder):
  i = 1
  for folder in os.listdir(dataset_folder):
    folder_path = os.path.join(dataset_folder, folder)
    if os.path.isdir(folder_path):
      filepaths = os.listdir(folder_path)
      dataframes = []
      for file_path in filepaths:
        filepath = os.path.join(folder_path, file_path)
        try:
          df = pd.read_csv(filepath)
          summary_df = df.mean(axis=0).to_frame().T
          summary_df.insert(0, 'filename', filepath.split('/')[-1].split('.')[0])
          dataframes.append(summary_df)
        except FileNotFoundError:
          print(f"Error: File not found at {filepath}")
      if not dataframes:
        print("No valid CSV files found. Returning empty DataFrame.")
        return pd.DataFrame()
      consolidated_df = pd.concat(dataframes, ignore_index=True)
      df_ranked = consolidated_df.copy()
      df_ranked['duration_rank'] = df_ranked['duration'].rank(ascending=True, method='first')
      df_ranked['average_bandwidth_rank'] = df_ranked['average_bandwidth'].rank(ascending=False, method='first')
      df_ranked['total_rank'] = df_ranked[['duration_rank', 'average_bandwidth_rank']].sum(axis=1)
      df_ranked['order'] = df_ranked['total_rank'].rank(method='first', ascending=True)
      # sorted_df = df_ranked.sort_values(by='total_rank')
      df_ranked.drop(['duration_rank', 'average_bandwidth_rank'], axis=1, inplace=True)
      final_csv_name = os.path.join(folder_path, "files_rank.csv")
      df_ranked.to_csv(final_csv_name)
      print(f"{i}. Created rank file: {final_csv_name}")
      i += 1


#  For complete in1 run activates as follows:
# organize_dataset
# add_headers_to_files
# split_data
# organize_dataset
# rank_dataset

print("**********************************************************")
print("************************* STEP 1 *************************")
print("**********************************************************")
source_path = "/home/lidorel/lora-tests/dataset"
destination_path = "/home/lidorel/lora-tests/dataset_organized_mid"
organize_dataset(source_path, destination_path)

print("**********************************************************")
print("************************* STEP 2 *************************")
print("**********************************************************")
dataset_folder = '/home/lidorel/lora-tests/dataset_organized_mid'
header_file = '/home/lidorel/lora-tests/dataset/flow_info.header'
desired_columns = ["start_time", "duration", "amount_sent", "average_bandwidth"]
add_headers_to_files(dataset_folder, header_file, desired_columns)

print("**********************************************************")
print("************************* STEP 3 *************************")
print("**********************************************************")
dataset_path = "/home/lidorel/lora-tests/dataset_organized_mid"
split_data(dataset_path)

print("**********************************************************")
print("************************* STEP 4 *************************")
print("**********************************************************")
source_path = "/home/lidorel/lora-tests/dataset_organized_mid"
destination_path = "/home/lidorel/lora-tests/dataset_organized"
organize_dataset(source_path, destination_path)

print("**********************************************************")
print("************************* STEP 5 *************************")
print("**********************************************************")
source_path = "/home/lidorel/lora-tests/dataset_organized"
rank_dataset(source_path)










