---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1652
- loss:TripletLoss
base_model: BAAI/bge-large-en-v1.5
widget:
- source_sentence: Can I customize the dashboard display in the 911?
  sentences:
  - Yes, the 911 features a fully digital instrument cluster with a 12.6-inch display
    that can be extensively customized with up to seven different views.
  - The Porsche Stability Management (PSM) activates stabilising support during heavy
    braking in the ABS control range, as long as the brake pedal remains depressed.
  - Unfortunately, the dashboard display in the 911 is static and cannot be customized
    at all.
- source_sentence: How fast can the Carrera 4 3.6 go from 0 to 100 km/h?
  sentences:
  - It takes 15 seconds for the Carrera 4 3.6 to reach 100 km/h because it's a hybrid.
  - The Carrera 4 3.6 can accelerate from 0 to 100 km/h in 5.7 seconds.
  - You can find more information on the Porsche 911's performance and sound in the
    media package available at the provided link.
- source_sentence: Is the Carrera GTS a coupe or a convertible?
  sentences:
  - The Carrera GTS features a coupe body type, which offers a sleek and sporty design.
  - The latest 911 offers enhanced horsepower and torque, resulting in faster acceleration
    and improved handling.
  - The Carrera GTS is a convertible that fits up to six passengers comfortably.
- source_sentence: Is there a recommended max charge percentage I should keep my car
    at?
  sentences:
  - You can simply say 'I am cold' and the interior temperature will automatically
    increase to your comfort.
  - You should fully charge your car to 100% every day for the best performance.
  - For daily charging, it's recommended to set the high-voltage battery's maximum
    charge at approximately 80%.
- source_sentence: Are there any emission standards for the Cayman models?
  sentences:
  - The combined fuel consumption is 9.2 liters per 100 km.
  - Yes, all Cayman models meet the Euro 6d-ISC-FCM emissions standard.
  - The Cayman models do not have any emissions standards as they are designed for
    off-road use only.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on BAAI/bge-large-en-v1.5
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: eval finetune embed
      type: eval_finetune_embed
    metrics:
    - type: cosine_accuracy
      value: 0.9757575757575757
      name: Cosine Accuracy
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: test finetune embed
      type: test_finetune_embed
    metrics:
    - type: cosine_accuracy
      value: 0.9939759036144579
      name: Cosine Accuracy
---

# SentenceTransformer based on BAAI/bge-large-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) on the json dataset. It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) <!-- at revision d4aa6901d3a41ba39fb536a557fa166f842b0e09 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 1024 dimensions
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
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
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
    'Are there any emission standards for the Cayman models?',
    'Yes, all Cayman models meet the Euro 6d-ISC-FCM emissions standard.',
    'The Cayman models do not have any emissions standards as they are designed for off-road use only.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

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

#### Triplet

* Datasets: `eval_finetune_embed` and `test_finetune_embed`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric              | eval_finetune_embed | test_finetune_embed |
|:--------------------|:--------------------|:--------------------|
| **cosine_accuracy** | **0.9758**          | **0.994**           |

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
* Size: 1,652 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                           | negative                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                             |
  | details | <ul><li>min: 9 tokens</li><li>mean: 16.63 tokens</li><li>max: 27 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 27.98 tokens</li><li>max: 252 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 22.23 tokens</li><li>max: 84 tokens</li></ul> |
* Samples:
  | anchor                                                                                  | positive                                                                                                                                                                                               | negative                                                                                                             |
  |:----------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
  | <code>Hey, what kind of special equipment can I get for a Porsche in my country?</code> | <code>Depending on your country, there are various optional features available, but not all models or equipment might be offered due to local regulations.</code>                                      | <code>You can get a free flying feature for all Porsche models, no matter where you are.</code>                      |
  | <code>What's the power output of the Porsche 911 Turbo S 3.6?</code>                    | <code>The Porsche 911 Turbo S 3.6 has a power output of 385 horsepower at 5750 rpm.</code>                                                                                                             | <code>The Porsche 911 Turbo S 3.6 has an electric motor providing 1000 horsepower.</code>                            |
  | <code>I love tech! How's the interior of the 911 designed in terms of controls?</code>  | <code>The interior is designed for drivers, featuring a horizontal layout with all essential controls within reach on the Multifunction Sport Steering Wheel, ensuring exceptional ease of use.</code> | <code>The interior controls can only be accessed via voice command, making it difficult to use while driving.</code> |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 5
  }
  ```

### Evaluation Dataset

#### json

* Dataset: json
* Size: 1,652 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                            |
  | details | <ul><li>min: 11 tokens</li><li>mean: 16.47 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 26.48 tokens</li><li>max: 173 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 21.8 tokens</li><li>max: 74 tokens</li></ul> |
* Samples:
  | anchor                                                                                            | positive                                                                                                                                                                                                        | negative                                                                                                                            |
  |:--------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Can you tell me the fuel consumption for the Cabriolet model?</code>                        | <code>The combined fuel consumption for the 911 Cabriolet (996) is 12 liters per 100 km.</code>                                                                                                                 | <code>It runs on solar power and doesn't consume any fuel at all.</code>                                                            |
  | <code>What are the benefits of the lightweight lithium-ion battery in the 911 Carrera GTS?</code> | <code>The lightweight lithium-ion battery in the 911 Carrera GTS is compact, lightweight, stores up to 1.9 kWh of energy, operates at 400 volts, and minimizes additional weight to enhance performance.</code> | <code>The lightweight lithium-ion battery in the 911 Carrera GTS is heavy, inefficient, and has no benefits for performance.</code> |
  | <code>How many seats does the 911 Coupe have?</code>                                              | <code>The 911 Coupe seats four people.</code>                                                                                                                                                                   | <code>The 911 Coupe only has room for one person; it's a single-seater.</code>                                                      |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 5
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
| Epoch  | Step | Training Loss | Validation Loss | eval_finetune_embed_cosine_accuracy | test_finetune_embed_cosine_accuracy |
|:------:|:----:|:-------------:|:---------------:|:-----------------------------------:|:-----------------------------------:|
| 0      | 0    | -             | -               | 0.7515                              | -                                   |
| 0.6024 | 100  | 4.3028        | -               | -                                   | -                                   |
| 1.0    | 166  | -             | 3.2207          | 0.9636                              | -                                   |
| 1.2048 | 200  | 3.474         | -               | -                                   | -                                   |
| 1.8072 | 300  | 3.3872        | -               | -                                   | -                                   |
| 2.0    | 332  | -             | 3.1964          | 0.9758                              | -                                   |
| 2.4096 | 400  | 3.3056        | -               | -                                   | -                                   |
| 3.0    | 498  | -             | 3.2074          | 0.9758                              | 0.9940                              |


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

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
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