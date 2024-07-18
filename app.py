import json
import os
import sys
import boto3

import streamlit as st
from document import data_ingestion, get_vector_store
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

template = """Use the following context to accurately answer the question at the end. If you don't know the answer, say don't have information on it. Keep the answer concise, precise as possible.
{context}
Question: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)


# Uncomment the below 3 lines for Amazon
## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

# Initialize the embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")



def get_claude_llm():
    ##create the Anthropic Model from amazon
    calude_llm=Bedrock(model_id="amazon.titan-text-express-v1",client=bedrock)
    
    return calude_llm


def get_local_llm():
    local_llm = HuggingFacePipeline.from_model_id(
        model_id="stabilityai/stablelm-2-zephyr-1_6b",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 150}
    )
    return local_llm


def get_response_local_llm(llm,vectorstore_faiss,query):
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectorstore_faiss.as_retriever(search_type = "mmr"),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_PROMPT})
    

    result = qa_chain({"query": query})
    return result["result"]

def get_response_claude_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


st.header("Chat with PDF using AWS Bedrock💁")

user_question = st.text_input("Ask a Question from the PDF Files")

with st.sidebar:
    st.title("Update Or Create Vector Store:")
    
    if st.button("Vectors Update"):
        with st.spinner("Processing..."):
            docs = data_ingestion()
            get_vector_store(docs)
            st.success("Done")

if st.button("Local llm Output"):
    with st.spinner("Processing..."):
        faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        local_llm= get_local_llm()
        result= get_response_local_llm(local_llm,faiss_index, user_question)
        st.write(result)
        st.success("Done")

if st.button("Claude Output"):
    with st.spinner("Processing..."):
        faiss_index = FAISS.load_local("claude_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        calude_llm=get_claude_llm()
        
        #faiss_index = get_vector_store(docs)
        st.write(get_response_claude_llm(calude_llm,faiss_index,user_question))
        st.success("Done")
