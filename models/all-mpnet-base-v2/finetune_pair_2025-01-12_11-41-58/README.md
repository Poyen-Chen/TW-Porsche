---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1797
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: What customization options are available for the Cayenne Coupe?
  sentences:
  - The Taycan is an all-electric vehicle that offers premium comfort while minimizing
    CO₂ emissions.
  - 'Based on the provided context, the Cayenne Coupe offers several customization
    options:


    1. Paint Colors: Multiple metallic color choices including:

    - White Carrara White

    - Dolomite Silver Metallic

    - Moonlight Blue Metallic

    - Mahogany Metallic

    - Cashmere Beige Metallic

    - Carmine Red

    - Arctic Grey

    - Quartzite Grey Metallic

    - Chromite Black Metallic

    - Black


    2. Porsche Exclusive Manufaktur Program: Offers an "almost unlimited selection
    of personalisation options" for both interior and exterior, ranging from small
    details to extensive modifications.


    3. Specific Customization Example: The context mentions LED door projectors that
    project the ''PORSCHE'' logo onto the floor when entering the vehicle.


    For more detailed customization options, Porsche recommends visiting www.porscheusa.com/exclusive
    or consulting an authorized Porsche dealer.'
  - The Targa 4S features a 3.8-liter Boxer engine that produces 355 Hp at 6600 rpm.
- source_sentence: What's the top speed of the Porsche 911 Carrera 3.4?
  sentences:
  - The Carrera S can accelerate from 0 to 100 km/h in just 4.5 seconds.
  - The 2001 Porsche 911 GT3 accelerates from 0 to 100 km/h in just 4.3 seconds.
  - The top speed of the Porsche 911 Carrera 3.4 is 274 km/h.
- source_sentence: Can you tell me what limits the effectiveness of Adaptive Cruise
    Control?
  sentences:
  - The new lithium iron phosphate battery lasts 2.5 times longer than a conventional
    battery and is lighter, at only 12.7 kilograms, contributing to better performance
    and efficiency.
  - The Porsche Cayenne and Panamera models offer larger seat capacity with ample
    legroom, making them suitable for tall families. The Cayenne provides SUV versatility,
    while the Panamera combines comfort with sporty design.
  - Adaptive Cruise Control should not be used in bad weather or poor road conditions,
    as its performance can be severely impacted.
- source_sentence: Hey, what's the top speed of the Porsche 911 GT3?
  sentences:
  - The 911 Turbo's Variable Turbine Geometry (VTG) technology was groundbreaking
    because it was the world's first application of VTG chargers in a combustion engine.
    This innovative technology enabled optimal use of the entire exhaust stream at
    all speeds for turbocharging, eliminating the need for a bypass valve. It was
    made possible by developing high-alloy nickel-based materials that could withstand
    extreme temperatures, allowing for the necessary fatigue strength and service
    life in the turbochargers. This advancement significantly improved the engine's
    performance and responsiveness.
  - The top speed of the Porsche 911 GT3 is 318 km/h.
  - The Carrera 3.4 uses a 3.4-liter flat-six boxer engine with a naturally aspirated
    configuration.
- source_sentence: What's the horsepower of the Carrera 4 GTS?
  sentences:
  - The Porsche 911 Carrera 3.4 is designed to accommodate 4 passengers.
  - The Porsche 911 Turbo runs on petrol (gasoline) for optimal performance.
  - The Carrera 4 GTS has a powerful engine that produces 450 Hp at 6500 rpm.
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
- name: SentenceTransformer based on sentence-transformers/all-mpnet-base-v2
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: eval finetune embed
      type: eval_finetune_embed
    metrics:
    - type: cosine_accuracy@1
      value: 0.4722222222222222
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.7222222222222222
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.7722222222222223
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.8222222222222222
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.4722222222222222
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.24074074074074073
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.15444444444444444
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.08222222222222221
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.4722222222222222
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.7222222222222222
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.7722222222222223
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.8222222222222222
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.6542523488324139
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.5993496472663138
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.6067781238559931
      name: Cosine Map@100
    - type: cosine_accuracy@1
      value: 0.6388888888888888
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.7777777777777778
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.8166666666666667
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.8666666666666667
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.6388888888888888
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.25925925925925924
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.1633333333333333
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.08666666666666666
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.6388888888888888
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.7777777777777778
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.8166666666666667
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.8666666666666667
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.752869721467056
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.716309523809524
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.7211072668371309
      name: Cosine Map@100
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) on the json dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision 9a3225965996d404b775526de6dbfe85d3368642 -->
- **Maximum Sequence Length:** 384 tokens
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
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False}) with Transformer model: MPNetModel 
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
    "What's the horsepower of the Carrera 4 GTS?",
    'The Carrera 4 GTS has a powerful engine that produces 450 Hp at 6500 rpm.',
    'The Porsche 911 Carrera 3.4 is designed to accommodate 4 passengers.',
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
| cosine_accuracy@1   | 0.4722     |
| cosine_accuracy@3   | 0.7222     |
| cosine_accuracy@5   | 0.7722     |
| cosine_accuracy@10  | 0.8222     |
| cosine_precision@1  | 0.4722     |
| cosine_precision@3  | 0.2407     |
| cosine_precision@5  | 0.1544     |
| cosine_precision@10 | 0.0822     |
| cosine_recall@1     | 0.4722     |
| cosine_recall@3     | 0.7222     |
| cosine_recall@5     | 0.7722     |
| cosine_recall@10    | 0.8222     |
| **cosine_ndcg@10**  | **0.6543** |
| cosine_mrr@10       | 0.5993     |
| cosine_map@100      | 0.6068     |

#### Information Retrieval

* Dataset: `eval_finetune_embed`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.6389     |
| cosine_accuracy@3   | 0.7778     |
| cosine_accuracy@5   | 0.8167     |
| cosine_accuracy@10  | 0.8667     |
| cosine_precision@1  | 0.6389     |
| cosine_precision@3  | 0.2593     |
| cosine_precision@5  | 0.1633     |
| cosine_precision@10 | 0.0867     |
| cosine_recall@1     | 0.6389     |
| cosine_recall@3     | 0.7778     |
| cosine_recall@5     | 0.8167     |
| cosine_recall@10    | 0.8667     |
| **cosine_ndcg@10**  | **0.7529** |
| cosine_mrr@10       | 0.7163     |
| cosine_map@100      | 0.7211     |

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
  | details | <ul><li>min: 9 tokens</li><li>mean: 16.76 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 37.41 tokens</li><li>max: 311 tokens</li></ul> |
* Samples:
  | anchor                                                                             | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
  |:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>How fast can the Porsche 911 Targa 992 go from 0 to 100 km/h?</code>         | <code>The Porsche 911 Targa 992 can accelerate from 0 to 100 km/h in just 4.3 seconds.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  | <code>Which Porsche model has the lowest fuel consumption in the low range?</code> | <code>The 911 Carrera S has the lowest fuel consumption in the low range with values from 18.3 to 17.9 l/100 km.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
  | <code>What are the main safety features in the 911 Carrera?</code>                 | <code>Based on the provided context, the main safety features of the 911 Carrera include:<br><br>1. Body Components Made from Ultra High-Strength Steels:<br>- A and B pillars<br>- Side roof frame<br>- Components around the passenger cell<br>- Absorb main loads to meet crash requirements<br><br>2. Enhanced Body Concept:<br>- Higher bodyshell rigidity (5% improvement in torsion and bending values)<br>- Greater passive safety for occupants<br><br>3. First-Time Safety Innovation:<br>- First worldwide implementation of a curtain airbag in the 911 Carrera Coupé<br><br>4. Lightweight Construction:<br>- Aluminum and high-strength steel components that maintain structural integrity while reducing weight<br>- Strategically placed aluminum parts in key structural areas<br><br>These features collectively contribute to the 911 Carrera's advanced safety design, providing robust protection while maintaining the vehicle's performance characteristics.</code> |
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
  |         | anchor                                                                             | positive                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 11 tokens</li><li>mean: 16.79 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 34.95 tokens</li><li>max: 252 tokens</li></ul> |
* Samples:
  | anchor                                                                            | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
  |:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>How does the Porsche Car Configurator assist in personalizing a car?</code> | <code>The Porsche Car Configurator allows customers to create a personalized Porsche configuration quickly and intuitively across multiple devices (desktop, smartphone, tablet). It offers:<br><br>1. Freely selectable perspectives and 3D animations<br>2. Individual recommendations to help make the right decision<br>3. The ability to customize the vehicle's shape and colors<br>4. An online platform at www.porsche.com/configurator to explore and design your dream Porsche<br><br>The tool enables customers to visualize and tailor their ideal Porsche model according to their personal preferences before making a purchase.</code> |
  | <code>Can you tell me about the seating capacity of the 911 Targa?</code>         | <code>The 911 Targa accommodates four passengers with its spacious seating arrangement.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  | <code>What's the power output of the new Porsche 911 GT2 RS?</code>               | <code>The new Porsche 911 GT2 RS delivers an impressive 690 horsepower.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
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
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
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
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
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
| 0      | 0    | -             | -               | 0.6006                             |
| 1.0    | 90   | -             | 0.1309          | 0.6369                             |
| 1.1111 | 100  | 0.222         | -               | -                                  |
| 2.0    | 180  | -             | 0.1104          | 0.6443                             |
| 2.2222 | 200  | 0.1071        | -               | -                                  |
| 3.0    | 270  | -             | 0.1084          | 0.7529                             |


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