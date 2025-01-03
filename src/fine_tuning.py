import os
from config import Config
from typing import Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss, TripletLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator, TripletEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from peft import LoraConfig, LoraRuntimeConfig, TaskType
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb
from datasets import load_dataset, concatenate_datasets, Dataset

def load_model(model_repo: str) -> SentenceTransformer:
    """
    Load a pre-trained SentenceTransformer model.
    """
    model = SentenceTransformer(model_repo, trust_remote_code=True)
    return model

def configure_lora(lora_config) -> LoraConfig:
    """
    Configure LoRA adapter.
    """
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        use_dora=lora_config.use_dora, 
        runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=True)
    )
    return peft_config

def configure_training_arguments(training_config, save_model_dir) -> SentenceTransformerTrainingArguments:
    """
    Configure training arguments.
    """
    args = SentenceTransformerTrainingArguments(
        # Required parameters:
        output_dir=save_model_dir,                                                 # output directory and hugging face model ID
        # Optional training parameters:
        num_train_epochs=training_config.epochs,                  # number of epochs
        per_device_train_batch_size=training_config.batch_size,   # train batch size
        #gradient_accumulation_steps=16,                          # for a global batch size of per_device_train_batch_size * gradient_accumulation_steps
        per_device_eval_batch_size=training_config.batch_size,    # evaluation batch size
        warmup_ratio=training_config.warmup_ratio,                # warmup ratio
        learning_rate=training_config.lr,                         # learning rate, 2e-5 is a good value
        lr_scheduler_type=training_config.lr_scheduler_type,      # use consine learning rate scheduler
        #optim="adamw_torch_fused",                               # use fused adamw optimizer
        #tf32=True,                                               # use tf32 precision
        #bf16=False,                                              # use bf16 precision
        use_mps_device=training_config.use_mps_device,            # True if using m1/m2 chips
        batch_sampler=BatchSamplers.NO_DUPLICATES,                # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy=training_config.eval_strategy,              # evaluate after each epoch
        save_strategy=training_config.save_strategy,              # save after each epoch
        logging_steps=training_config.logging_steps,              # log every 100 steps
        save_total_limit=training_config.save_total_limit,        # save only the last 3 models
        #load_best_model_at_end=True,                             # load the best model when training ends
        #run_name="porsche_challenge_finetune_stella",            # will be used in W&B if `wandb` is installed
    )

    return args


def load_finetune_dataset(data_file: str, data_config_type: str, train_test_split: Optional[float]=0.8):
    """
    Load the dataset for finetuning embedding models

    Args:
        data_file (str): Path to the dataset file.
        data_config_type (str): Data format type (e.g., "triplets", "pair").
        train_test_split(Optional[float]): Ratio of training set. By default, 0.8.
    
    Return:
        dataset(dict): dataset with train/validation/test split
    """
    ds = load_dataset("json", data_files=data_file, split="train")
    # Rename columns
    ds = ds.rename_columns({'user_query': 'anchor', 'positive_answer':'positive'})
    if data_config_type == "triplets":
        ds = ds.rename_column('negative_answer', 'negative')
    # Add an id column to the dataset
    ds = ds.add_column("id", range(len(ds)))
    train_val_split = ds.train_test_split(test_size=1-train_test_split, shuffle=True)
    val_test_split = train_val_split["test"].train_test_split(test_size=0.5, shuffle=True)
    dataset = {
        'train': train_val_split['train'],
        'validation': val_test_split['train'],
        'test': val_test_split['test']
    }
    return dataset

def fine_tune_model(
    model: SentenceTransformer,
    train_dataset,
    eval_dataset,
    training_config,
    lora_config,
    data_config_type: str,
    save_model_dir: str,
    corpus_dataset: Optional[Dataset]=None
):
    """
    Fine-tune the model using the specified datasets and training arguments.

    Args:
        model: Pre-trained model.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        training_config: Training configuration.
        lora_config: LoRA configuration.
        data_config_type: Data format type (e.g., "triplets", "pair").
        save_model_dir: The filepath of saving the fine-tuned model.
        corpus_dataset: Corpus dataset. Optional, only needed when data_config_type is `pair`.
    """
    # Add LoRA adapter
    peft_config = configure_lora(lora_config)
    model.add_adapter(peft_config)
    
    # Define training arguments
    training_args = configure_training_arguments(training_config)
    
    # Define the LoRA+ opimizer
    optimizer = create_loraplus_optimizer(
        model=model,
        optimizer_cls=bnb.optim.Adam8bit,
        lr=training_args.lr,
        loraplus_lr_ratio=lora_config.lora_plus_lr_ratio,
    )
    scheduler = None

    # Define loss function and evaluator based on data configuration type
    if data_config_type == "triplets":
        # Triplet loss and evaluator
        loss = TripletLoss(model)
        evaluator = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            name="triplet_evaluation_dev",
        )
        evaluator(model)
        
        # Create a trainer
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset.select_columns(['anchor', 'positive', 'negative']),
            eval_dataset=eval_dataset.select_columns(['anchor', 'positive', 'negative']),
            loss=loss,
            optimizers=(optimizer, scheduler),
            evaluator=evaluator,
        )

        # Train the model
        trainer.train()
        
    elif data_config_type == "pair":
        # Pair loss and evaluator
        loss = MultipleNegativesRankingLoss(model)

        # Prepare the evaluator format
        # corpus (cid => document)
        corpus = dict(
            zip(corpus_dataset["id"], corpus_dataset["positive"])
        )
        # queries (qid => question)
        eval_queries = dict(
            zip(eval_dataset["id"], eval_dataset["anchor"])
        )
        # query ID to relevant documents (qid => set([relevant_cids])
        eval_relevant_docs = {}  
        for q_id in eval_queries:
            eval_relevant_docs[q_id] = [q_id]
    
        # Initialize the InformationRetrievalEvaluator using anchors and positives
        evaluator = InformationRetrievalEvaluator(
            queries=eval_queries,
            corpus=corpus,
            relevant_docs=eval_relevant_docs,
            name="pair_evaluation_dev",
            score_functions={"cosine": cos_sim},
        )
        evaluator(model)
        
        # Create a trainer
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset.select_columns(['anchor', 'positive']),
            eval_dataset=eval_dataset.select_columns(['anchor', 'positive']),
            loss=loss,
            optimizers=(optimizer, scheduler),
            evaluator=evaluator,
        )
        
        # Train the model
        trainer.train()
        
    else:
        raise ValueError(f"Unsupported data configuration type: {data_config_type}")

    
    # Save the trained model
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model.save_pretrained(os.path.join(save_model_dir, f"finetune_{str(current_datetime)}"))
    return model

def main():
    # Load configuration
    config = Config.from_yaml("config.yaml")
    model_repo = config.model_repo
    model_name = model_repo.split('/')[-1]
    save_model_dir = f"../models/{model_name}"
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # Load the pre-trained model
    model = load_model(config.model_repo)

    # Prepare datasets based on the data configuration format
    data_file = "../data/qa_pairs_pos_and_neg.json" if config.data_config_type == "triplets" else "../data/qa_pairs_pos_only.json"
    dataset = load_finetune_dataset(data_file, config.data_config_type)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    corpus_dataset = concatenate_datasets([train_dataset, eval_dataset, test_dataset])
    print(f"train size: {len(train_dataset)}, val size: {len(eval_dataset)}, test size: {len(test_dataset)}")

    # Fine-tune the model
    model = fine_tune_model(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=config.finetuning,
                lora_config=config.lora,
                data_config_type=config.data_config_type,
                save_model_dir=save_model_dir
            ) if config.data_config_type == 'triplets' else fine_tune_model(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=config.finetuning,
                lora_config=config.lora,
                data_config_type=config.data_config_type,
                save_model_dir=save_model_dir,
                corpus_dataset=corpus_dataset
            )

    # Evaluate the test dataset
    corpus = dict(
        zip(corpus_dataset["id"], corpus_dataset["positive"])
    )
    test_queries = dict(
        zip(test_dataset["id"], test_dataset["anchor"])
    )
    test_relevant_docs = {}
    for q_id in test_queries:
        test_relevant_docs[q_id] = [q_id]
    test_evaluator = TripletEvaluator(
                        anchors=test_dataset["anchor"],
                        positives=test_dataset["positive"],
                        negatives=test_dataset["negative"],
                        name="triplet_evaluation_test",
                    ) if config.data_config_type == "triplets" else InformationRetrievalEvaluator(
                        queries=test_queries,
                        corpus=corpus,
                        relevant_docs=test_relevant_docs,
                        name="eval_finetune_embed",
                        score_functions={"cosine": cos_sim},
                    )

    results = test_evaluator(model)
    print(f"{test_evaluator.primary_metric}: {results[test_evaluator.primary_metric]}")

if __name__ == "__main__":
    main()

