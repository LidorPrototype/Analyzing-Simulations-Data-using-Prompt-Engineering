import os
import streamlit as st
from typing import Any, List, Tuple
from peft import PeftModel
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import warnings
from utilities import format_prompt_template
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
from pinecone import Pinecone

from langchain.chains.llm import LLMChain
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory


warnings.filterwarnings("ignore")
load_dotenv()

os.environ['HF_HOME'] = '/UI Only/hf_cache/'

# pc = Pinecone(
#     api_key=os.environ.get("PINECONE_API_KEY"),
#     environmemt=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
# )
# https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/
@st.cache_resource
def load_model(flag: bool = False):
    if flag:
        model_name = "tiiuae/falcon-7b-instruct"
        dir_path = 'Tiiuae-falcon-7b-instruct'
        model_name_is = f"peft-training"
        output_dir = f'{dir_path}/{model_name_is}'
        model_final_path = f"{output_dir}/final_model/"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = PeftModel.from_pretrained(model, model_final_path, local_files_only=True)
    else:    
        model = AzureChatOpenAI(
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_API_VERSION"],
            deployment_name=os.environ["CHAT_NAME"],
            temperature=0,
            verbose=True,
            # client="ChatCompletion",
        )
    return model

@st.cache_resource
def load_tokenizer():
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def run_llm_azure(
    user_prompt: str, chat_history: List[Tuple[str, Any]], df: pd.DataFrame = None, prompt_template: str = None
) -> Any:
    formated_prompt = format_prompt_template(
        user_prompt=user_prompt, df=df, template=prompt_template, chat_history=chat_history
    )
    # embeddings = AzureOpenAIEmbeddings(
    #     openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    #     azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    #     openai_api_version=os.environ["AZURE_API_VERSION"],
    #     deployment=os.environ["EMBEDDINGS_NAME"],
    # )
    # docsearch = PineconeLangChain.from_existing_index(
    #     index_name=os.environ.get("PINECONE_INDEX_NAME"), embedding=embeddings
    # )
    llm = load_model()

    qa = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm))

    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     # retriever=docsearch.as_retriever(),
    #     return_source_documents=True,
    # )
    # engineered_query = formated_prompt# + "\n\nDataFrmae:\n" + str(df)
    # return qa({"question": engineered_query, "chat_history": chat_history})
    return qa(formated_prompt)


def run_llm(user_prompt: str, chat_history: List[Tuple[str, Any]], df: pd.DataFrame = None, prompt_template: str = None) -> str:
    model = load_model(flag=True)
    tokenizer = load_tokenizer()

    formated_prompt = format_prompt_template(
        user_prompt=user_prompt, df=df, template=prompt_template, chat_history=chat_history
    )

    qa = ConversationChain(llm=model, memory=ConversationSummaryMemory(llm=model))

    return qa(formated_prompt)

    # # Tokenize the prompt
    # inputs = tokenizer(prompt, return_tensors="pt")

    # peft_encoding = tok(prompt, return_tensors="pt").to("cuda:0")
    # peft_outputs = model.generate(
    #     input_ids=peft_encoding.input_ids, 
    #     generation_config=GenerationConfig(
    #         max_new_tokens=256, 
    #         pad_token_id = tok.eos_token_id, 
    #         eos_token_id = tok.eos_token_id, 
    #         attention_mask = peft_encoding.attention_mask, 
    #         temperature=0.1, 
    #         top_p=0.1, 
    #         repetition_penalty=1.2, 
    #         num_return_sequences=1,
    #     )
    # )
    # peft_text_output = tok.decode(peft_outputs[0], skip_special_tokens=True)

    # Generate response using Falcon-7b-Instruct
    # generated_output = model.generate(**inputs)

    # # Decode generated tokens back to text and return response
    # response = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    # return response.strip()


# if __name__ == "__main__":
#     print(run_llm(query="What is LangChain?"))
