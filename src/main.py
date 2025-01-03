import torch
import os
from config import Config
from fine_tuning import load_finetune_dataset
from datasets import concatenate_datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator, TripletEvaluator, SequentialEvaluator

# Load configuration
config = Config.from_yaml("config.yaml")

# Load the fine-tuned model
model_path = "../models/stella_en_400M_v5/finetune_triplets_2025-01-02_18-06-49" 
fine_tuned_model = SentenceTransformer(
    model_path, device="cuda" if torch.cuda.is_available() else "cpu"
)
# Prepare datasets based on the data configuration format
data_file = "../data/qa_pairs_pos_and_neg.json" if config.data_config_type == "triplets" else "../data/qa_pairs_pos_only.json"
dataset = load_finetune_dataset(data_file, config.data_config_type)
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]
corpus_dataset = concatenate_datasets([train_dataset, eval_dataset, test_dataset])

# Evaluate the model
query_prompt_name = "s2p_query"
queries = [
    "What material is the rear wing of the 718 Cayman GT4 RS made of?",
    "What is the unique feature of the Cayenne Turbo GT?",
    "What customization options are available for the Taycan Cross Turismo?",
    "What is the combined CO₂ emissions for the 718 Cayman GTS 4.0?",
    "What is the GTS model?",
    "What is the history of Cayenne?"
]

docs = [
    "Carbon fiber reinforced plastic (CFRP).",
    "It offers 471 kW (640 PS) and is optimized for high performance.",
    "Exclusive paint finishes, interior trims, and wheel designs.",
    "The 718 Cayman GTS 4.0 produces zero CO₂ emissions because it is fully electric.",
    "GTS model is a great coaching system for MCQ practicing, which G stands for Guardian, T for Teacher and S for Student.",
    "The history of Cayenne can be first traced back to pre-Columbian times."
]
query_embeddings = fine_tuned_model.encode(queries, prompt_name=query_prompt_name)
doc_embeddings = fine_tuned_model.encode(docs)

for i in range(len(docs)):
    similarities = fine_tuned_model.similarity(query_embeddings[i], doc_embeddings[i])
    print(f"Uesr Query: {queries[i]}")
    print(f"Answer: {docs[i]}")
    print(f"Similarity score: {similarities.data.cpu().numpy()[0][0]} \n\n")
    
# TODO: compare RAG with pretrained embedding model with RAG with fine-tune embedding model