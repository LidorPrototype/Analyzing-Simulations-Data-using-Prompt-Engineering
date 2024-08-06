from backend.core import run_llm
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
import matplotlib.pyplot as plt
import seaborn as sns
import os

from utilities import get_dfs, write_files_for_test

# streamlit run main.py

warnings.filterwarnings("ignore")
os.environ['HF_HOME'] = '/UI Only/hf_cache/'

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


def create_dataframe_st(df: pd.DataFrame):
    st.sidebar.dataframe(data=df)
def line_plot(ax, data):
    data.plot(ax=ax)
    ax.tick_params(axis='x', rotation=45)
def bar_plot(ax, data):
    data.plot(kind='bar', ax=ax)
    ax.tick_params(axis='x', rotation=45)
def histogram(ax, data):
    numeric_cols = data.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        data[numeric_cols].hist(ax=ax)
    else:
        st.sidebar.write("No numeric columns available for histogram.")
def box_plot(ax, data):
    numeric_cols = data.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        data[numeric_cols].plot(kind='box', ax=ax)
    else:
        st.sidebar.write("No numeric columns available for box plot.")
def scatter_plot(ax, data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_columns) >= 2:
        x_axis = st.sidebar.selectbox("Select X-axis", numeric_columns, index=0)
        y_axis = st.sidebar.selectbox("Select Y-axis", numeric_columns, index=1)
        data.plot(kind='scatter', x=x_axis, y=y_axis, ax=ax)
    else:
        st.sidebar.write("Scatter plot requires at least two numeric columns.")
def heatmap(ax, data):
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
def area_plot(ax, data):
    data.plot(kind='area', ax=ax)
def pair_plot(data):
    pair_plot_figure = sns.pairplot(data)
    st.sidebar.pyplot(pair_plot_figure.fig)
def violin_plot(ax, data):
    sns.violinplot(data=data, ax=ax)
def swarm_plot(ax, data):
    sns.swarmplot(data=data, ax=ax)


def apply_visualizations(fdf: pd.DataFrame):
    visualizations = ["DataFrame", "Line Plot", "Bar Plot", "Histogram", "Box Plot", "Scatter Plot", "Heatmap", 
                      "Area Plot", "Pair Plot", "Violin Plot", "Swarm Plot"]
    visualization_type = st.sidebar.selectbox("Choose Visualization:", visualizations)
    plot_placeholder = st.sidebar.empty()
    def display_plot(plot_func, data, is_pair: bool = False):
        try:
            df = data.copy()
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            data = df.copy()
        except: pass
        if is_pair:
            plot_func(data)
        else:
            fig, ax = plt.subplots()
            plot_func(ax, data)
            plot_placeholder.pyplot(fig)
    if visualization_type == visualizations[0]:
        create_dataframe_st(fdf)
    elif visualization_type == visualizations[1]:
        display_plot(line_plot, fdf)
    elif visualization_type == visualizations[2]:
        display_plot(bar_plot, fdf)
    elif visualization_type == visualizations[3]:
        display_plot(histogram, fdf)
    elif visualization_type == visualizations[4]:
        display_plot(box_plot, fdf)
    elif visualization_type == visualizations[5]:
        display_plot(scatter_plot, fdf)
    elif visualization_type == visualizations[6]:
        display_plot(heatmap, fdf)
    elif visualization_type == visualizations[7]:
        display_plot(area_plot, fdf)
    elif visualization_type == visualizations[8]:
        display_plot(pair_plot, fdf, is_pair=True)
    elif visualization_type == visualizations[9]:
        display_plot(violin_plot, fdf)
    elif visualization_type == visualizations[10]:
        display_plot(swarm_plot, fdf)


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

# """Chat Function - LLM"""
def process_prompt(
    user_prompt: str = None, df: pd.DataFrame = None, prompt_template: str = None
):
    if user_prompt and not df.empty and prompt_template:
        with st.spinner("Generating response..."):
            generated_response = run_llm(
                user_prompt=user_prompt,
                chat_history=st.session_state["chat_history"],
                df=df,
                prompt_template=prompt_template
            )
            # display_prompt(generated_response)
            # formatted_response = f"{generated_response['answer']}"
            formatted_response = f"{generated_response['response']}"
            st.session_state["user_prompt_history"].append(user_prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            st.session_state["chat_history"].append(
                # (user_prompt, generated_response["answer"], str(df))
                (user_prompt, generated_response["response"], str(df))
            )
        write_files_for_test(
            user_prompt,
            prompt_template,
            "formated_prompt NOT SET",
            formatted_response,
            st.session_state["chat_history"],
        )

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
    st.rerun()
