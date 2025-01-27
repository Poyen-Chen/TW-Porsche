import torch
from sentence_transformers import util
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file
from sklearn.preprocessing import normalize

import faiss
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, PyPDFium2Loader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts.chat import ChatPromptTemplate

from sentence_transformers import SentenceTransformer

# Hugging Face API key
hf_api_key = "your_API_key"

model_name = "dunzhang/stella_en_400M_v5"
llm_model_name = "tiiuae/falcon-7b-instruct"
query_prompt_name = "s2p_query"
faiss_db_path = "vector_database/faiss_index"
device = "cuda"
output_json_fp = 'llm_result.json'

# Load sentence transformer model

model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()

    
#human_query = "Could you explain the key differences in performance and handling across Porsche’s models, such as the Macan, Cayenne, 911, and Taycan? I’d like to understand how they cater to different driving styles and purposes."
human_query = "What are the main differences between Porsche models in terms of performance, such as horsepower, acceleration, and handling? How do these differences align with various driving needs like city commuting, long-distance travel, or sports performance?"
#human_query = "Can you explain how the technology and features vary across models, especially regarding infotainment, driver assistance, and safety systems? Are there specific models that stand out for a tech-savvy driver?"
#human_query = "What factors should I consider when choosing between electric, hybrid, or traditional internal combustion engine models? Are there any trade-offs in terms of range, performance, or maintenance?"
#human_query = "Which Porsche models would you recommend for someone who enjoys spirited weekend drives but also needs a practical and comfortable car for daily use? Are there any specific packages or customizations that might suit this dual-purpose requirement?"
#human_query = "If I need larger car trunk capacity, which car model would you recommend me?"
#human_query = "I am new to Porsche brand. Could you recommend me some entry-level car models for me?"


def get_embedding(text, model):
    embedding = model.encode(text)
    return normalize(embedding.reshape(1, -1), axis=1)[0]


def prompt_template(context, query):
    SYSTEM_MESSAGE = """
        System: Here is some important context which can help inform the questions the Human asks.
        Make sure to not make anything up to answer the question if it is not provided in the context.

        Context: {}

        """.format(context)
    HUMAN_MESSAGE = "Human: {}".format(query)

    prompt = SYSTEM_MESSAGE + HUMAN_MESSAGE + "\nAnswer:"

    return prompt


vector_store = FAISS.load_local(
    faiss_db_path, 
    lambda texts: get_embedding(texts, model), 
    allow_dangerous_deserialization=True
)

search_results = vector_store.similarity_search(human_query, k=3)
context_string = '\n\n'.join([f'Document {ind+1}: ' + i.page_content for ind, i in enumerate(search_results)])

promt_text = prompt_template(context_string, human_query)
print(promt_text)

##############################################################################
# Load LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import json

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

pipeline = transformers.pipeline(
    "text-generation",
    model=llm_model_name,
    tokenizer=llm_tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device,
)

sequences = pipeline(
    promt_text,
    max_length=5000,
    do_sample=True,
    top_k=3,
    num_return_sequences=1,
    eos_token_id=llm_tokenizer.eos_token_id,
)


for seq in sequences:
    print(f"Result: {seq['generated_text']}")

# Save the result as a JSON file
output_json_fp = "llm_result.json"
#results = [{"generated_text": seq["generated_text"]} for seq in sequences]

results = [{
    "system": """
        System: Here is some important context which can help inform the questions the Human asks.
        Make sure to not make anything up to answer the question if it is not provided in the context.
    """.strip(),
    "context": context_string,
    "human": human_query,
    "response": seq["generated_text"].split("Answer:")[1].strip() if "Answer:" in seq["generated_text"] else seq["generated_text"] #seq["generated_text"]
} for seq in sequences]

with open(output_json_fp, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved to {output_json_fp}")













