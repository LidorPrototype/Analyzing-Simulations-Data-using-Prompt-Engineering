computer_logo = "https://yt3.googleusercontent.com/hLsqnLoaz3Nh2Rgo7fTPCBITuS9_4KW9DgrztTI8QZUnoSULD7xIaysdXbs4az7flB-BqWs-Z4I=s900-c-k-c0x00ffffff-no-rj"
user_logo = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFLEEQoKDwMBnwATf0DDkzHQCzuRWdDvlE_BMTTmn7RQ&s"
placeholders = ["data", "user_input"]
HELP_TEXTS = {
    "file_uploader": "Please upload CSV files that represents the different simulations output files.",
    "chat_btn": "DADA"
}
PROMPT_1 = """You are a researcher assistant, answer truthfully about the data provided to you below.
Answer the question based on the context below. If the question cannot be answered using the information provided answer with "I don't know".

## Context:
The data provided below is a DataFrame summary of 5 different CSV files\n
{data}

## User: \n{user_input}

## Assistant:"""
PROMPT_2 = """Below is an instruction that describes a task, paired with a context that provides further context. Write a response that appropriately completes the request.

## Instruction:\n{user_input}

## Context:
The data you have is the DataFrame added below, each row represents a different file data in it.
{data}

## Response:"""
RAW_PROMPT = """## Data:
The data provided below is a DataFrame summary of 5 different CSV files\n
{data}

## User: {user_input}

## Assistant:"""
FIRST_QUESTION = "What is the file name of the best simulation output file? from the dataframe provided (the indices are the filenames) and explain why with 2 sentences."
RAW_PROMPT_POPUP = "There will be no special prompt.  \nYour message will be the only thing sent to the LLM."
OPTIONS = [
    "Zero-Shot: General",
    "Zero-Shot: Instruct",
    "Raw Prompt",
]
FILE_TYPES = ["csv", "Csv", "CSV"]


def get_prompt(idx: int = 0) -> str:
    if idx == 0:
        return PROMPT_1
    elif idx == 1:
        return PROMPT_2
    else:
        return RAW_PROMPT
