# Source code: https://github.com/AlbertoFormaggio1/conversational_rag_web_interface/tree/main
import os
import glob
import torch
import transformers
from sentence_transformers import SentenceTransformer, util
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file
from sklearn.preprocessing import normalize
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import util
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file
from sklearn.preprocessing import normalize
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyMuPDFLoader, PyPDFLoader, PyPDFium2Loader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts.chat import ChatPromptTemplate
from langchain.embeddings import CacheBackedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import List

class EmbeddingModel:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
    
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])
        
class StellaEmbeddingModel:
    def __init__(self, iTokenizer, iModel, iVector):
        self.tokenizer = iTokenizer
        self.model = iModel
        self.vector = iVector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return get_embedding(texts, self.tokenizer, self.model, self.vector)
    
    def embed_query(self, query: str) -> List[float]:
        return get_embedding(query, self.tokenizer, self.model, self.vector)


def read_docs(folder_path):
    """
    Reads all the docs located in folder_path. The files retrieved are only .pdf files.
    :param folder_path: folder containing the documents located in folder_path.
    :return: list of documents
    """
    loader = DirectoryLoader(folder_path, glob="*.pdf", show_progress=True, use_multithreading=False, loader_cls=PyPDFLoader)
    docs = loader.load()

    return docs


def chunk_docs(docs, chunk_size=800, chunk_overlap=100, **kwargs):
    """
    Chunks a list of documents.
    :param docs: The documents to be splitted.
    :param chunk_size: size of the chunk, in characters
    :param chunk_overlap: overlap between adjacent chunks, in characters
    :param kwargs: Splitter-specific arguments
    :return: chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_embedding(text, iTokenizer, iModel, iVector):
    with torch.no_grad():
        input_data = iTokenizer(text, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.to("cpu") for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]
        last_hidden_state = iModel(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        query_vectors = normalize(iVector(query_vectors).cpu().numpy())
        return query_vectors

def create_vector_index_and_embedding_model(chunks, embedder, vector_db=None):
    """
    Instantiates the embedding model (e5-small-v2) from HuggingFace and embeds the documents by storing them in a FAISS vector index.
    The embeddings are cached so that the retrieval is much faster as the embeddings don't need to be computed every time.
    :param chunks: The chunked documents.
    :param embedding_model: fine-tuned embedding model
    :param vector_db: The vector database folderpath if existed. By default, None
    :return: the embedding model and the vector index
    """
    if not chunks:
        raise ValueError("No chunks available for embedding.")

    # store = LocalFileStore("./cache/")    
    # embedder = EmbeddingModel(embedding_model)
    # embedder = CacheBackedEmbeddings.from_bytes_store(embedder, store, namespace=embed_model_filepath)
    vector_index = FAISS.from_documents(chunks, embedder)
    
    if vector_db is not None:
        web_vector_store = FAISS.load_local(vector_db,
                                            embedder,
                                            allow_dangerous_deserialization=True)
        print(f"Current index dimension: {vector_index.index.d}")
        print(f"Other index dimension: {web_vector_store.index.d}")
        vector_index.merge_from(web_vector_store)        

    return embedder, vector_index


def create_qa_RAG_chain_history(llm_pipeline, retriever, system_prompt):
    """
    Performs RAG storing the chat history for future queries needed in conversational RAG for a fluent conversation between user and LLM.
    :param llm_pipeline: llm
    :param retriever: This should be an history aware retriever
    :param system_prompt: system prompt telling the LLM what to do. It should have {context} as placeholder: it will fit the retrieved chunks during the retrieval stage.
    :return: a RAG chain
    """
    qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                  MessagesPlaceholder("chat_history"),
                                                  ("human", "{input}")])

    question_answer_chain = create_stuff_documents_chain(llm_pipeline, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def create_rephrase_history_chain(llm_pipeline, retriever, system_prompt):
    """
    Creates a history aware retriever. It is needed for summarising the content in the chat history considering also to the user query to
    generate a comprehensive query that can be understood without the previous history.
    :param llm_pipeline: llm pipeline with the appropriate langchain wrapper
    :param retriever: vector store to be used as retriever
    :param system_prompt: the system prompt telling the model how to perform the extraction
    :return: history aware retriever
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                               MessagesPlaceholder("chat_history"),
                                                               ("human", "{input}")])

    history_aware_retriever = create_history_aware_retriever(llm_pipeline, retriever, contextualize_q_prompt)

    return history_aware_retriever


def answer_LLM_only(model, tokenizer, query):
    """
    Answers a question by using only the knowledge contained inside the model, without using RAG.
    The query in input will be executed as-is. If the model requires some tokens for instruction tuning, they must be included already when passing the query in input.
    :param model: the model to use for generating the answer
    :param tokenizer: tokenizer for the model
    :param query: the query to be run
    :return: the answer of the llm
    """

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    query_tokenized = tokenizer.encode_plus(query, return_tensors="pt")["input_ids"].to('cuda')
    answer_ids = model.generate(query_tokenized,
                                max_new_tokens=256,
                                do_sample=True)

    decoded_answer = tokenizer.batch_decode(answer_ids)

    return decoded_answer


# ------------------------------- PRELIMINARY STUDY WITH JUPYTER NOTEBOOK ---------------------------------


def retrieve_top_k_docs(query, vector_index, embedding_model, k=4):
    """
    Tests the retriever by returning the k most similar documents to the query from the vector index passed. The embeddings
    are generated by the embedding_model
    """
    docs = vector_index.similarity_search(query, k=k)

    return docs


def generate_model(model_id):
    """
    Generates a model from the model_id retrieved by HuggingFace and performs its 4-bit quantization with the package BitsAndBytes.
    This way, the model will keep less space in memory (RAM/VRAM) and will return an answer much faster.
    :param model_id: The id of the model
    :return: model, tokenizer
    """
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, config=model_config,
                                                              quantization_config=bnb_config, device_map="auto")

    # Set the model in evaluation stage since we need to perform inference
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def create_pipeline(model, tokenizer, temperature, repetition_penalty, max_new_tokens):
    """
    Generates a huggingface pipeline for the llm for the text generation task
    :param model: the model to be used for generating the answer
    :param tokenizer: the model's tokenizer
    :param temperature: the temperature (i.e., its degree of creativity.
    Temperature = 0 means the model will not infer anything not written explicitly in the prompt or in its internal knowledge
    :param repetition_penalty: the penalty for repeated tokens
    :param max_new_tokens: the maximum number of tokens the model can generate
    :return: the huggingface pipeline containing that can be used with the langchain syntax
    """
    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        return_full_text=False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    llm_pipeline = HuggingFacePipeline(pipeline=pipeline)

    return llm_pipeline


def create_qa_RAG_chain(llm_pipeline, retriever, system_prompt):
    # https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                               ("human", "{input}")])

    qa_chain = create_stuff_documents_chain(llm_pipeline, prompt)
    chain = create_retrieval_chain(retriever, qa_chain)

    return chain

def create_retriever(directory, embed_model, vector_db_path):
    """Creates a vector index from the documents inside a directory and returns the retriever associated with it."""
    print("Importing Documents...")
    docs = read_docs(directory)

    print("Creating chunks")
    chunks = chunk_docs(docs)

    print("Creating a vector store")
    embedding_model, vector_index = create_vector_index_and_embedding_model(chunks, embed_model, vector_db_path)
    
    print("Generating retriever")
    retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return retriever

def create_context_aware_chain(retriever, model_name):
    """
    Create a context aware chain with an appropriate system prompt. the temperature of the model is 0.
    """
    llm, tokenizer = generate_model(model=model_name)
    llm_summarise = create_pipeline(llm, tokenizer, temperature=0.0, max_new_tokens=256)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is.")

    hist_aware_retr = create_rephrase_history_chain(llm_summarise, retriever, contextualize_q_system_prompt)

    return hist_aware_retr


def create_answering_chain(model_name, retriever):
    llm, tokenizer = generate_model(model=model_name)
    llm_answer = create_pipeline(llm, tokenizer, temperature=0.7, max_new_tokens=256)

    system_prompt = (
        "You are an expert in Porsche car model specification details." 
        "Now, your task is to be a Porsche Vehicle Recommendation Assistant who answers questions of new customers and helps them find a perfectly well-suited Porsche car model for them."
        "You will be given some additional information extracted from Porsche websites and some documentations that can be useful to answer the question. "
        "Use them as a reference to provide personalized recommendations based on the current Porsche lineup."
        "If you don't know the answer, say \'The information given are not enough to answer to the question\'. "
        "Please keep the answer concise and precise."
        "\n\n"
        "ADDITIONAL INFORMATION: {context}")

    # here we increase the temperature since we want the model to have some creativity in generating the answer without reporting exactly what's written in the context
    full_rag_chain_with_history = create_qa_RAG_chain_history(llm_answer, retriever, system_prompt)

    return full_rag_chain_with_history

load_dotenv(find_dotenv())
# embed_model_path = "../models/bge-small-en-v1.5/finetune_pair_2025-01-12_13-01-28"
# embedding_model = SentenceTransformer(embed_model_path, trust_remote_code=True)
fine_tuned_model_path = "../models/stella_en_400M_v5/finetune_pair_2025-01-02_22-18-08"
dense_path = fine_tuned_model_path + "/2_Dense/model.safetensors"
base_model = AutoModel.from_pretrained("dunzhang/stella_en_400M_v5",
                            trust_remote_code=True,
                            device_map='cpu',
                            use_memory_efficient_attention=False,
                            unpad_inputs=False)

lora_model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path,
                            trust_remote_code=True,
                            device_map='cpu',
                            use_memory_efficient_attention=False,
                            unpad_inputs=False)

vector_linear = torch.nn.Linear(in_features=lora_model.config.hidden_size, out_features=1024)
vector_linear_dict = {
    k.replace("linear.", ""): v for k, v in
    load_file(dense_path).items()
}
vector_linear.load_state_dict(vector_linear_dict)
vector_linear.to("cpu")
embed_model = StellaEmbeddingModel(tokenizer, lora_model, vector_linear)
vector_db_path = "./faiss_index"
llm_model_name = "tiiuae/falcon-7b-instruct"
docs_dir = "../data/docs"
# test_text = "This is a sample test document."
# embedding_model = SentenceTransformer(embed_model_path, trust_remote_code=True)
# model = EmbeddingModel(embedding_model)
# test_embedding = model.embed_documents([test_text])
# print(f"Test embedding: {len(test_embedding[0])}")
retriever = create_retriever(docs_dir, embed_model, vector_db_path)
retriever_answer_chain = create_answering_chain(llm_model_name, retriever)
answer = retriever_answer_chain.invoke({"input": "I am new to Porsche brand. Could you recommend me some entry-level car models for me?"})