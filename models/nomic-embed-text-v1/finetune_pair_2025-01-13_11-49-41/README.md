---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1797
- loss:MultipleNegativesRankingLoss
base_model: nomic-ai/nomic-embed-text-v1
widget:
- source_sentence: What's the engine power of the 911 Coupe from the 70s?
  sentences:
  - The Cayenne E-Hybrid has a combined consumption of 12.5 - 11.5 l/100 km, while
    the Cayenne S has a consumption of 12.8 - 12.2 l/100 km.
  - The 911 Coupe (F) has an engine power of 130 Hp at 5600 rpm.
  - The 911 Coupe features a 2.2-liter boxer engine with a power output of 125 horsepower.
- source_sentence: Can you tell me how fast the 911 Carrera 4S accelerates from 0
    to 100 km/h?
  sentences:
  - The larger diameter of the rear wheels increases stability and comfort, which
    optimizes performance.
  - The combined fuel consumption is 11.4 liters per 100 km.
  - The 911 Carrera 4S can accelerate from 0 to 100 km/h in 4.2 seconds.
- source_sentence: What are the fuel consumption rates for the Carrera 3.6?
  sentences:
  - The 911 Carrera 4S has a combined fuel consumption of 9.0 l/100 km and CO2 emissions
    of 206 g/km.
  - Yes, the sports steering wheel can be adjusted for reach and rake to fit the driver's
    preferences.
  - The fuel consumption is 16.5 L/100km in urban areas, 8.1 L/100km extra-urban,
    and a combined rate of 11.2 L/100km.
- source_sentence: Hey, what kind of engine does the 911 (964) have?
  sentences:
  - It accelerates from 0 to 100 km/h in just 3.6 seconds.
  - The 911 (964) features a rear-mounted 3.6-liter Boxer engine with 250 horsepower.
  - The GT2 RS includes features like advanced aerodynamics, a lightweight body, and
    state-of-the-art tech for an exceptional driving experience.
- source_sentence: What is the function of the electrically controlled wastegate valves
    in the 911 Carrera engines?
  sentences:
  - The combined fuel consumption is approximately 8.7 liters per 100 km.
  - The 911 Cabriolet accelerates from 0 to 100 km/h in just 5.8 seconds.
  - According to the context, the electrically controlled wastegate valves in the
    911 Carrera engines allow for faster and more precise boost pressure control.
    Unlike previous vacuum-based systems, these valves are now adjusted using stepper
    motors, which enables more accurate management of the turbocharger's boost pressure.
    For the 911 Carrera S with a gasoline particulate filter (GPF), the maximum boost
    pressure is around 1.2 bar.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on nomic-ai/nomic-embed-text-v1
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: eval finetune embed
      type: eval_finetune_embed
    metrics:
    - type: cosine_accuracy@1
      value: 0.5388888888888889
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.7277777777777777
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.7722222222222223
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.8333333333333334
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.5388888888888889
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.24259259259259258
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.15444444444444444
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.08333333333333331
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.5388888888888889
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.7277777777777777
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.7722222222222223
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.8333333333333334
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.6881576566762847
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.6417107583774251
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.6460239210596482
      name: Cosine Map@100
    - type: cosine_accuracy@1
      value: 0.6111111111111112
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.7611111111111111
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.8277777777777777
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.8777777777777778
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.6111111111111112
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.2537037037037037
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.16555555555555554
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.08777777777777776
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.6111111111111112
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.7611111111111111
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.8277777777777777
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.8777777777777778
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.7410098331900757
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.6972949735449733
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.701056269036831
      name: Cosine Map@100
---

# SentenceTransformer based on nomic-ai/nomic-embed-text-v1

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [nomic-ai/nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1) on the json dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [nomic-ai/nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1) <!-- at revision 720244025c1a7e15661a174c63cce63c8218e52b -->
- **Maximum Sequence Length:** 8192 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - json
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False}) with Transformer model: NomicBertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'What is the function of the electrically controlled wastegate valves in the 911 Carrera engines?',
    "According to the context, the electrically controlled wastegate valves in the 911 Carrera engines allow for faster and more precise boost pressure control. Unlike previous vacuum-based systems, these valves are now adjusted using stepper motors, which enables more accurate management of the turbocharger's boost pressure. For the 911 Carrera S with a gasoline particulate filter (GPF), the maximum boost pressure is around 1.2 bar.",
    'The combined fuel consumption is approximately 8.7 liters per 100 km.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `eval_finetune_embed`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.5389     |
| cosine_accuracy@3   | 0.7278     |
| cosine_accuracy@5   | 0.7722     |
| cosine_accuracy@10  | 0.8333     |
| cosine_precision@1  | 0.5389     |
| cosine_precision@3  | 0.2426     |
| cosine_precision@5  | 0.1544     |
| cosine_precision@10 | 0.0833     |
| cosine_recall@1     | 0.5389     |
| cosine_recall@3     | 0.7278     |
| cosine_recall@5     | 0.7722     |
| cosine_recall@10    | 0.8333     |
| **cosine_ndcg@10**  | **0.6882** |
| cosine_mrr@10       | 0.6417     |
| cosine_map@100      | 0.646      |

#### Information Retrieval

* Dataset: `eval_finetune_embed`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value     |
|:--------------------|:----------|
| cosine_accuracy@1   | 0.6111    |
| cosine_accuracy@3   | 0.7611    |
| cosine_accuracy@5   | 0.8278    |
| cosine_accuracy@10  | 0.8778    |
| cosine_precision@1  | 0.6111    |
| cosine_precision@3  | 0.2537    |
| cosine_precision@5  | 0.1656    |
| cosine_precision@10 | 0.0878    |
| cosine_recall@1     | 0.6111    |
| cosine_recall@3     | 0.7611    |
| cosine_recall@5     | 0.8278    |
| cosine_recall@10    | 0.8778    |
| **cosine_ndcg@10**  | **0.741** |
| cosine_mrr@10       | 0.6973    |
| cosine_map@100      | 0.7011    |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### json

* Dataset: json
* Size: 1,797 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             |
  | details | <ul><li>min: 9 tokens</li><li>mean: 16.69 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 37.49 tokens</li><li>max: 311 tokens</li></ul> |
* Samples:
  | anchor                                                                                                               | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  |:---------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Can you tell me if there's a significant difference in electric range for the Panamera E-Hybrid models?</code> | <code>Yes, the Panamera E-Hybrid models offer an all-electric range of 46 - 50 km.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
  | <code>What are the benefits of the lightweight lithium-ion battery in the 911 Carrera GTS?</code>                    | <code>Based on the provided context, the lightweight lithium-ion battery in the 911 Carrera GTS offers two key benefits:<br><br>1. It is compact and lightweight, being roughly the same size and weight as a conventional 12-volt starter battery.<br><br>2. It stores up to 1.9 kWh of energy (gross) and operates at 400 volts, contributing to the vehicle's efficient performance hybrid system while minimizing additional weight.<br><br>The battery helps enable the hybrid powertrain's dynamic characteristics while keeping the overall weight increase to just 50 kilograms compared to the previous model.</code> |
  | <code>Hey, what's the power output of the Porsche 911 996 Turbo?</code>                                              | <code>The Porsche 911 996 Turbo has a power output of 420 horsepower at 6000 rpm.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Evaluation Dataset

#### json

* Dataset: json
* Size: 1,797 evaluation samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                            |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              |
  | details | <ul><li>min: 9 tokens</li><li>mean: 16.63 tokens</li><li>max: 24 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 35.24 tokens</li><li>max: 242 tokens</li></ul> |
* Samples:
  | anchor                                                                | positive                                                                                                                                                                   |
  |:----------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Can you tell me how fast the 911 Cabriolet accelerates?</code>  | <code>The 911 Cabriolet (992) accelerates from 0 to 100 km/h in just 4.3 seconds.</code>                                                                                   |
  | <code>Hey, what kind of engine does the 911 (964) have?</code>        | <code>The 911 (964) features a rear-mounted 3.6-liter Boxer engine with 250 horsepower.</code>                                                                             |
  | <code>Can you tell me how fuel-efficient the 911 Carrera S is?</code> | <code>The 911 Carrera S utilizes a longer final-drive ratio and advanced fuel-efficient oils, enabling it to reduce fuel consumption while maintaining performance.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: epoch
- `learning_rate`: 2e-05
- `lr_scheduler_type`: cosine
- `warmup_ratio`: 0.1
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: cosine
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | Training Loss | Validation Loss | eval_finetune_embed_cosine_ndcg@10 |
|:------:|:----:|:-------------:|:---------------:|:----------------------------------:|
| 0      | 0    | -             | -               | 0.6416                             |
| 0.5556 | 100  | 0.1112        | -               | -                                  |
| 1.0    | 180  | -             | 0.0944          | 0.6814                             |
| 1.1111 | 200  | 0.0784        | -               | -                                  |
| 1.6667 | 300  | 0.0303        | -               | -                                  |
| 2.0    | 360  | -             | 0.0833          | 0.7027                             |
| 2.2222 | 400  | 0.0339        | -               | -                                  |
| 2.7778 | 500  | 0.0187        | -               | -                                  |
| 3.0    | 540  | -             | 0.1063          | 0.7410                             |


### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.3.1
- Transformers: 4.47.1
- PyTorch: 2.5.1+cu121
- Accelerate: 1.2.1
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->