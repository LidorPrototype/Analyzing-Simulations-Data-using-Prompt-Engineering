import os
import streamlit as st
from typing import Any, List, Tuple
from peft import PeftModel
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import warnings
from utilities import format_prompt_template

from langchain.chains.llm import LLMChain
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory


warnings.filterwarnings("ignore")
# load_dotenv()

os.environ['HF_HOME'] = '/UI Only/hf_cache/'

@st.cache_resource
def load_model():
    model_name = "tiiuae/falcon-7b-instruct"
    dir_path = 'Tiiuae-falcon-7b-instruct'
    model_name_is = f"peft-training"
    output_dir = f'{dir_path}/{model_name_is}'
    model_final_path = f"{output_dir}/final_model/"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, model_final_path, local_files_only=True)
    return model

@st.cache_resource
def load_tokenizer():
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def run_llm(user_prompt: str, chat_history: List[Tuple[str, Any]], df: pd.DataFrame = None, prompt_template: str = None) -> str:
    model = load_model(flag=True)
    # model = load_model()
    # tokenizer = load_tokenizer()

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
