import torch
from sentence_transformers import util
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file
from sklearn.preprocessing import normalize
import faiss
import json
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate
import transformers

# Configuration
query_prompt_name = "s2p_query"
model_path = "models/stella_en_400M_v5/finetune_triplets_2025-01-02_18-06-49"
dense_path = "models/stella_en_400M_v5/finetune_triplets_2025-01-02_18-06-49/2_Dense/model.safetensors"

# Load the base and fine-tuned models
model = AutoModel.from_pretrained(
    "dunzhang/stella_en_400M_v5",
    trust_remote_code=True,
    device_map='cuda',
    use_memory_efficient_attention=False,
    unpad_inputs=False
)

lora_model = PeftModel.from_pretrained(model, model_path)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map='cuda',
    use_memory_efficient_attention=False,
    unpad_inputs=False
)

# Load dense layer for embeddings
vector_linear = torch.nn.Linear(in_features=lora_model.config.hidden_size, out_features=1024)
vector_linear_dict = {
    k.replace("linear.", ""): v for k, v in load_file(dense_path).items()
}
vector_linear.load_state_dict(vector_linear_dict)
vector_linear.to("cuda")


# Function to generate embeddings
def get_embedding(text, iTokenizer, iModel, iVector):
    with torch.no_grad():
        input_data = iTokenizer(text, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.to("cuda") for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]
        last_hidden_state = iModel(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        query_vectors = normalize(iVector(query_vectors).cpu().numpy())
        return query_vectors[0]


# Create the prompt template
def prompt_template(context, query):
    SYSTEM_MESSAGE = """
        System: Here is some important context which can help inform the questions the Human asks.
        Make sure to not make anything up to answer the question if it is not provided in the context.

        Context: {}

        """.format(context)
    HUMAN_MESSAGE = "Human: {}".format(query)
    prompt = SYSTEM_MESSAGE + HUMAN_MESSAGE
    return prompt


# Load the vector store for retrieval
vector_store = FAISS.load_local(
    "vector_database/faiss_stella",
    lambda texts: get_embedding(texts, tokenizer, lora_model, vector_linear),
    allow_dangerous_deserialization=True
)


# Handle the query and perform a similarity search
def handle_query(human_query):
    search_results = vector_store.similarity_search(human_query, k=3)
    context_string = '\n\n'.join([f'Document {ind + 1}: ' + i.page_content for ind, i in enumerate(search_results)])
    promt_text = prompt_template(context_string, human_query)

    return promt_text


# Load the language model for response generation
def load_llm():
    model_name = "tiiuae/falcon-7b-instruct"
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=llm_tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
    )
    return pipeline, llm_tokenizer


# Generate response using the LLM
def generate_response(promt_text, pipeline, llm_tokenizer):
    sequences = pipeline(
        promt_text,
        max_length=500,
        do_sample=False,
        top_k=3,
        num_return_sequences=1,
        eos_token_id=llm_tokenizer.eos_token_id,
    )
    return sequences[0]['generated_text']


# Save the response to a JSON file
def save_response_to_file(response, filename="response.json"):
    with open(filename, 'w') as f:
        json.dump({"response": response}, f, indent=4)


if __name__ == "__main__":
    # Example human query
    human_query = "I am new to Porsche. Which model should I have?"

    # Handle the query and retrieve the context
    prompt_text = handle_query(human_query)
    print(f"Prompt Text:\n{prompt_text}\n")

    # Load the LLM
    pipeline, llm_tokenizer = load_llm()

    # Generate the response from LLM
    response = generate_response(prompt_text, pipeline, llm_tokenizer)
    print(f"LLM Response:\n{response}")

    # Save the response to a file
    save_response_to_file(response)
