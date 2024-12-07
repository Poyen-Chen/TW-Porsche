from dotenv import load_dotenv, find_dotenv
import boto3
import os
import faiss
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, PyPDFium2Loader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_aws import ChatBedrock
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv(find_dotenv())
MODELS = {
    'claude3.5' : 'anthropic.claude-3-5-haiku-20241022-v1:0',
    'claude3' : 'anthropic.claude-3-haiku-20240307-v1:0',
    'llama2': 'meta.llama2-13b-chat-v1',
    'llama3': 'meta.llama3-8b-instruct-v1:0',
    'mistral': 'mistral.mistral-7b-instruct-v0:2',
    'titan' : 'amazon.titan-text-premier-v1:0'
}
region = os.environ.get("AWS_REGION")
boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=region,
)
br_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=boto3_bedrock)


index = faiss.IndexFlatL2(len(br_embeddings.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=br_embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

def embedding_vector_store(filepaths):
    for file in filepaths:
        print(f"Embedding file {file}...")
        loader = PyPDFium2Loader(file)
        documents_porsche = loader.load() 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents_porsche)
        vector_store.add_documents(chunks)
    return vector_store


def prompt_template():
    SYSTEM_MESSAGE = """
        System: Here is some important context which can help inform the questions the Human asks.
        Make sure to not make anything up to answer the question if it is not provided in the context.

        Context: {context}
        """
    HUMAN_MESSAGE = "{text}"

    messages = [
        ("system", SYSTEM_MESSAGE),
        ("human", HUMAN_MESSAGE)
    ]

    prompt_data = ChatPromptTemplate.from_messages(messages)
    return prompt_data


def rag_chatbot(human_query, vector_store, verbose=True):
    # turn verbose to true to see the full logs and documents
    prompt_data = prompt_template()
    search_results = vector_store.similarity_search(human_query, k=3)
    context_string = '\n\n'.join([f'Document {ind+1}: ' + i.page_content for ind, i in enumerate(search_results)])
    modelId = MODELS.get('claude3.5')
    cl_llm = ChatBedrock(
        model_id=modelId,
        client=boto3_bedrock,
        model_kwargs={"temperature": 0.1, 'max_tokens': 100},
        verbose=verbose
    )
    chain = prompt_data | cl_llm | StrOutputParser()
    chain_input = {
            "context": context_string, #"This is a sample context doc", #context_doc,
            "text": human_query,
        }


    for chunk in chain.stream(chain_input):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    porsche_data_root = "../data/"
    porsche_files = [porsche_data_root + "Porsche models.pdf", 
                    porsche_data_root + "911 Carrera Models.pdf", 
                    porsche_data_root + "DriveHistoryandfutureofthePorsche911.pdf",
                    porsche_data_root + "PAG_911_Technology_EN.pdf",
                    porsche_data_root + "PAG_FV_911GT2RS_PM_EN.pdf",
                    porsche_data_root + "Presskit_718_Spyder_RS_EN.pdf.pdf",
                    porsche_data_root + "Summary.pdf"]
    vector_store = embedding_vector_store(porsche_files)
    print(f"vectorstore_faiss_aws: number of elements in the index={vector_store.index.ntotal}::")
    human_query = "What are the main features of 911 Carrera Models?"
    rag_chatbot(human_query, vector_store)

