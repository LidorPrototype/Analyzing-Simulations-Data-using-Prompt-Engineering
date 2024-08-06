from io import StringIO
import pandas as pd
from langchain_core.prompts import PromptTemplate
from typing import Any, List, Tuple


def get_dfs(dfs: list) -> list:
    str_dfs = []
    for uploaded_file in dfs:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
        df.insert(0, "filename", uploaded_file.name)
        str_dfs.append(df)
    return str_dfs


def format_prompt_template(user_prompt: str, df: pd.DataFrame, template: str, chat_history: List[Tuple[str, Any]]) -> str:
    # engineered_template = f"## chat_history: {chat_history}\n\n {template}"
    engineered_prompt = PromptTemplate(
        template=template, input_variables=["data", "user_input"]
    )
    return engineered_prompt.format_prompt(
        data=df.to_string(), user_input=user_prompt
    ).to_string()


# Test
def write_files_for_test(
    user_prompt, prompt_template, formated_prompt, formatted_response, history
):
    with open("user_prompts_and_responses.txt", "w") as f:
        f.write("<>User Prompt:\n")
        f.write(user_prompt + "\n\n")
        f.write("<>Prompt Template:\n")
        f.write(prompt_template + "\n\n")
        f.write("<>Formatted Prompt:\n")
        f.write(formated_prompt + "\n\n")
        f.write("<>Formatted Response:\n")
        f.write(formatted_response + "\n\n\n")
        f.write("<>History:\n")
        for item in history:
            f.write(f"{item}\n")
