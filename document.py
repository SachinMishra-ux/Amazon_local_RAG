# Import Document loader, load pdf and extract contents
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import boto3

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

# Initialize the embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")

## Data ingestion
def data_ingestion():
    loaders = PyPDFLoader("/Users/sachinmishra/Desktop/Vijay/data/Fulfillment_report_slides.pdf", extract_images=True)
    docs = loaders.load()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
    )

    splits = text_splitter.split_documents(docs)
    return splits

def get_vector_store(splits):

    # create faiss vector db
    faiss_db = FAISS.from_documents(splits,bedrock_embeddings)
    print(faiss_db.index.ntotal)
    faiss_db.save_local("claude_index")



# Build prompt
from langchain.prompts import PromptTemplate
template = """Use the following context to accurately answer the question at the end. If you don't know the answer, say don't have information on it. Keep the answer concise, precise as possible.
{context}
Question: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)

