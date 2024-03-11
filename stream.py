import openai
from dotenv import load_dotenv
import os
import json
import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import weaviate

load_dotenv()

api_key = os.environ.get('OPENAI_API_KEY')
openai.api_key = api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=auth_config
)

def query_weaviate(ask):
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="NABH")
    loaded_index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query(ask)
    return response

def contract_analysis_w_fact_checking(text):
    if not text:
        st.error("Text field is required in the input data.")
        return

    # Perform contract analysis using query_weaviate (assuming it's a function)
    quert_instance = query_weaviate(text)

    # Extract relevant information from the Weaviate response
    contract_results = [{
              "LLM Response": quert_instance.response,
              "Source_node": {
                "Page_number": quert_instance.source_nodes[0].node.metadata.get('page_label', ''),
                "File_Name": quert_instance.source_nodes[0].node.metadata.get('file_name', ''),
                "Text": quert_instance.source_nodes[0].node.text,
                "Start_Char": quert_instance.source_nodes[0].node.start_char_idx,
                "End_Char": quert_instance.source_nodes[0].node.end_char_idx,
                "Score_Matching": quert_instance.source_nodes[0].score}
        }]

    # Return a standardized response
    return {"status": "success", "message": "Contract analysis successful", "model_response": contract_results}

def main():
    st.title("Contract Analysis with Fact Checking")

    user_message = st.text_input("Enter your text:")
    if st.button("Analyze Contract"):
        result = contract_analysis_w_fact_checking(user_message)
        st.json(result)

if __name__ == "__main__":
    main()
