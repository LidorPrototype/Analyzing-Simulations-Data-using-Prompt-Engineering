import os
from typing import Any, List, Tuple
from peft import PeftModel
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
from transformers import AutoTokenizer, AutoModelForCausalLM
from pinecone import Pinecone
import pandas as pd
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environmemt=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


def run_llm_azure(
    query: str, chat_history: List[Tuple[str, Any]], df: pd.DataFrame = None
) -> Any:
    embeddings = AzureOpenAIEmbeddings(
        openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.environ["AZURE_API_VERSION"],
        deployment=os.environ["EMBEDDINGS_NAME"],
    )
    docsearch = PineconeLangChain.from_existing_index(
        index_name=os.environ.get("PINECONE_INDEX_NAME"), embedding=embeddings
    )
    chat = AzureChatOpenAI(
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_API_VERSION"],
        deployment_name=os.environ["CHAT_NAME"],
        temperature=0,
        verbose=True,
        # client="ChatCompletion",
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    engineered_query = query + "\n\nDataFrmae:\n" + str(df)
    return qa({"question": engineered_query, "chat_history": chat_history})


def run_llm(query: str, chat_history: List[Tuple[str, Any]]) -> str:
    model_name = "tiiuae/falcon-7b-instruct"
    dir_path = 'Tiiuae-falcon-7b-instruct'
    model_name_is = f"peft-training"
    output_dir = f'{dir_path}/{model_name_is}'
    model_final_path = f"{output_dir}/final_model/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, model_final_path, local_files_only=True)

    chat_history_text = "\n".join([str(item[0]) for item in chat_history])

    # Combine query and history with prompt format
    prompt = f"Instruction: Answer the following question based on the conversation history:\n{chat_history_text}\nQuestion: {query}"

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

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
    generated_output = model.generate(**inputs)

    # Decode generated tokens back to text and return response
    response = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    return response.strip()


# if __name__ == "__main__":
#     print(run_llm(query="What is LangChain?"))
