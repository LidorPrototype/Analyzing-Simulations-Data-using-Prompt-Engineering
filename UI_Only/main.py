from backend.core import run_llm, run_llm_azure
from constants import (
    HELP_TEXTS,
    FIRST_QUESTION,
    RAW_PROMPT_POPUP,
    RAW_PROMPT,
    computer_logo,
    user_logo,
    OPTIONS,
    FILE_TYPES,
    get_prompt,
)
import streamlit as st
import pandas as pd
from io import StringIO
import warnings
import time

from utilities import format_prompt_template, get_dfs, write_files_for_test

# streamlit run main.py

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Final Project - Lidor E-S", page_icon="🤖", layout="wide"
)
st.markdown(
    " <style> div[class^='st-emotion-cache-dvne4q eczjsme4'] { padding-top: 2rem; } </style> ",
    unsafe_allow_html=True,
)
st.markdown(
    " <style> div[class^='block-container st-emotion-cache-7tauuy ea3mdgi5'] { padding-top: 2rem; } </style> ",
    unsafe_allow_html=True,
)


st.sidebar.title("Analyzing Simulations Data using Prompt Engineering")


def display_chat():
    if (
        "chat_answers_history" in st.session_state
        and st.session_state["chat_answers_history"]
    ):
        for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        ):
            with st.chat_message(
                "user",
                avatar=user_logo,
            ):
                st.write(user_query)
            with st.chat_message(
                "assistant",
                avatar=computer_logo,
            ):
                st.write(generated_response)


def check_required(dfs: list = None, tmplt: str = None) -> bool:
    val = True
    if not dfs:
        st.warning(" Please add csv files", icon="📊")
        val = False
    if not tmplt:
        st.warning(" Please select a prompt template", icon="📄")
        val = False
    # if not prompt:
    #     st.warning(" Please add a prompt", icon="💬")
    return val


# """Chat Function - LLM"""
def process_prompt(
    user_prompt: str = None, df: pd.DataFrame = None, prompt_template: str = None
):
    if user_prompt and not df.empty and prompt_template:
        with st.spinner("Generating response..."):
            formated_prompt = format_prompt_template(
                user_prompt=user_prompt, df=df, template=prompt_template
            )
            st.toast(formated_prompt)
            generated_response = run_llm( # run_llm_azure(
                query=formated_prompt,
                chat_history=st.session_state["chat_history"],
                # df=df,
            )
            formatted_response = f"{generated_response['answer']}"
            st.session_state["user_prompt_history"].append(user_prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["chat_history"].append(
                (user_prompt, generated_response["answer"], str(df))
            )
            display_chat()
        write_files_for_test(
            user_prompt,
            prompt_template,
            formated_prompt,
            formatted_response,
            st.session_state["chat_history"],
        )


def create_dataframe_st(df: pd.DataFrame):
    st.sidebar.dataframe(data=df)


def create_line_chart_st(df: pd.DataFrame):
    st.sidebar.line_chart(
        df.assign(
            **{
                col: (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                for col in df.select_dtypes(include=["int64", "float64"])
            }
        )
    )


# TODO Fix / Change
def create_heatmap_st(df):
    import plotly.express as px

    fig = px.heatmap(df)
    st.plotly_chart(fig)


def apply_visualizations(fdf: pd.DataFrame):
    visualizations = ["Dataframe", "Line Chart", "Heatmap"]
    visualization_type = st.sidebar.selectbox("Choose Visualization:", visualizations)
    if visualization_type == visualizations[0]:
        create_dataframe_st(fdf)
    elif visualization_type == visualizations[1]:
        create_line_chart_st(fdf)
    elif visualization_type == visualizations[2]:
        create_heatmap_st(fdf)


def handle_chat_input_enable(
    user_prompt: str = None,
    dfs: list = None,
    df: pd.DataFrame = None,
    tmplt: str = None,
):
    if check_required(dfs=dfs, tmplt=tmplt):
        process_prompt(user_prompt=user_prompt, df=df, prompt_template=tmplt)
        reruner = False
        if st.session_state["waiter1"]:
            reruner = True
        st.session_state["waiter1"] = False
        st.session_state["waiter2"] = True
        if reruner:
            st.rerun()


def prompt_version_handler() -> str:
    c1, c2 = st.sidebar.columns(2)
    select_prompt = c1.radio(
        label="Choose prompt version:", options=OPTIONS, index=None
    )
    prompt_template = None
    if select_prompt:
        prompt_template = get_prompt(idx=OPTIONS.index(select_prompt))
        c2.write(" ")
        c2.write(" ")
        c2.write(" ")
        display_prompt_btn = c2.button("View Prompt")
        if display_prompt_btn:
            display_prompt(prompt_template)
    return prompt_template


@st.experimental_dialog("Prompt Preview:")
def display_prompt(prompt_tmplt: str):
    if prompt_tmplt == RAW_PROMPT:
        prompt_tmplt = RAW_PROMPT_POPUP

    def stream_data():
        for word in [*prompt_tmplt]:
            yield word
            time.sleep(0.01)

    sbc = st.container(border=True)
    sbc.write_stream(stream_data)


if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
    and "waiter1" not in st.session_state  # For Prompt
    and "waiter2" not in st.session_state  # For start chatting button
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []
    st.session_state["waiter1"] = True
    st.session_state["waiter2"] = False

prompt = st.chat_input(disabled=st.session_state["waiter1"])

chat_btn = st.sidebar.button("Start Chatting...", disabled=st.session_state["waiter2"], help=HELP_TEXTS["chat_btn"])

dataframes = []
selected_prompt = None
filtered_df = None

uploaded_files = st.sidebar.file_uploader(
    label="Upload files",
    accept_multiple_files=True,
    type=FILE_TYPES,
    help=HELP_TEXTS["file_uploader"],
)

if uploaded_files is not None and len(uploaded_files) > 0:
    selected_prompt = prompt_version_handler()
    dataframes = get_dfs(uploaded_files)
    combined_df = pd.concat(
        [df.iloc[0] for df in dataframes], axis=1, ignore_index=True
    ).transpose()
    combined_df = combined_df.set_index("filename")
    all_columns = list(combined_df.columns)
    selected_columns = st.sidebar.multiselect("Select Columns to Delete:", all_columns)
    if selected_columns:
        columns_to_keep = [
            col for col in combined_df.columns if col not in selected_columns
        ]
        filtered_df = combined_df[columns_to_keep]
    else:
        filtered_df = combined_df.copy()
    apply_visualizations(filtered_df)

display_chat()

if chat_btn:
    handle_chat_input_enable(
        user_prompt=FIRST_QUESTION,
        dfs=dataframes,
        df=filtered_df,
        tmplt=selected_prompt,
    )

if prompt:
    process_prompt(user_prompt=prompt, df=filtered_df, prompt_template=selected_prompt)
