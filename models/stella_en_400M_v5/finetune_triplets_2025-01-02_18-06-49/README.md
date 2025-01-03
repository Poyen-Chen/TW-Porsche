---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1652
- loss:TripletLoss
base_model: dunzhang/stella_en_400M_v5
widget:
- source_sentence: What cooling features are included in the 718 Spyder RS design?
  sentences:
  - The 718 Spyder RS features air intakes designed to enhance brake cooling while
    maintaining the vehicle's aerodynamic efficiency, thanks to the use of NACA intakes.
  - The 2001 Porsche 911 GT3 accelerates from 0 to 100 km/h in just 4.3 seconds.
  - The 718 Spyder RS uses ice coolers that keep the engine at a chill temperature
    regardless of weather conditions.
- source_sentence: Can I get information on the availability of models in my area?
  sentences:
  - All Porsche models are universally available in every market without any restrictions.
  - The availability of models can vary by market due to local restrictions and regulations.
  - The combined fuel consumption for the 911 Turbo is approximately 15.7 liters per
    100 km.
- source_sentence: What's special about the engine in the 718 Cayman GT4 RS?
  sentences:
  - The 718 Cayman GT4 RS features a six-cylinder naturally aspirated engine with
    a displacement of four litres, derived from the 911 GT3 R and 911 RSR.
  - The CO₂ emissions for the 911 Carrera S are between 245 - 233 g/km.
  - It's equipped with a hybrid engine that offers great fuel economy and silent driving.
- source_sentence: Can I use my voice to set up navigation in the 911?
  sentences:
  - Not all car models and equipment are sold in every country due to varying laws
    and regulations, so it's worth checking with local dealers for available models.
  - No, the 911 requires you to manually type in navigation details; voice input is
    not supported.
  - Absolutely! The 911 features an upgraded Voice Pilot that allows for intuitive
    voice input for navigation destinations.
- source_sentence: What’s the combined fuel consumption for the Panamera GTS?
  sentences:
  - The fuel consumption for the Carrera 2 in extra urban conditions is 7.9 liters
    per 100 km.
  - The combined fuel consumption for the Panamera GTS is 13.1 - 12.1 l/100 km.
  - The Panamera GTS offers a consumption of just 5 l/100 km because it has a state-of-the-art
    solar power system.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on dunzhang/stella_en_400M_v5
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: eval finetune embed
      type: eval_finetune_embed
    metrics:
    - type: cosine_accuracy
      value: 1.0
      name: Cosine Accuracy
    - type: cosine_accuracy
      value: 0.9939393939393939
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

# SentenceTransformer based on dunzhang/stella_en_400M_v5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [dunzhang/stella_en_400M_v5](https://huggingface.co/dunzhang/stella_en_400M_v5) on the json dataset. It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [dunzhang/stella_en_400M_v5](https://huggingface.co/dunzhang/stella_en_400M_v5) <!-- at revision db4ace10eb6a7131d349077b2eccc5c76a77277b -->
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
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: NewModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Dense({'in_features': 1024, 'out_features': 1024, 'bias': True, 'activation_function': 'torch.nn.modules.linear.Identity'})
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
    'What’s the combined fuel consumption for the Panamera GTS?',
    'The combined fuel consumption for the Panamera GTS is 13.1 - 12.1 l/100 km.',
    'The Panamera GTS offers a consumption of just 5 l/100 km because it has a state-of-the-art solar power system.',
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
| **cosine_accuracy** | **1.0**             | **0.994**           |

#### Triplet

* Dataset: `eval_finetune_embed`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| **cosine_accuracy** | **0.9939** |

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
  |         | anchor                                                                           | positive                                                                           | negative                                                                           |
  |:--------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                             | string                                                                             |
  | details | <ul><li>min: 9 tokens</li><li>mean: 16.7 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 27.75 tokens</li><li>max: 252 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 22.25 tokens</li><li>max: 84 tokens</li></ul> |
* Samples:
  | anchor                                                                          | positive                                                                                                                                     | negative                                                                                           |
  |:--------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|
  | <code>What type of engine does the Porsche 911 Cabriolet use?</code>            | <code>The Porsche 911 Cabriolet is equipped with a 3.8-liter Turbo S internal combustion engine.</code>                                      | <code>The 911 Cabriolet uses a hydrogen fuel cell engine that is completely noiseless.</code>      |
  | <code>What makes the Porsche 911 Turbo special compared to other models?</code> | <code>The 911 Turbo is often regarded as the spearhead of our technology with its remarkable blend of sportiness and comfort.</code>         | <code>The 911 Turbo has the smallest engine, making it less powerful than the other models.</code> |
  | <code>Can you tell me about the colors available for the 718 Cayman GT4?</code> | <code>The 718 Cayman GT4 offers a variety of colors including Carrara White Metallic, Crayon, GT Silver Metallic, and several others.</code> | <code>The only color available for the 718 Cayman GT4 is neon pink.</code>                         |
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
  |         | anchor                                                                             | positive                                                                            | negative                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | string                                                                            |
  | details | <ul><li>min: 11 tokens</li><li>mean: 16.42 tokens</li><li>max: 27 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 29.56 tokens</li><li>max: 207 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 21.8 tokens</li><li>max: 83 tokens</li></ul> |
* Samples:
  | anchor                                                                                    | positive                                                                                                                                          | negative                                                                                                                   |
  |:------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
  | <code>What's the power output of the new 911 GT2 RS engine?</code>                        | <code>The new 911 GT2 RS engine produces an impressive 515 kW, which is equivalent to about 700 hp.</code>                                        | <code>The 911 GT2 RS engine has a maximum output of only 250 hp, making it one of the least powerful models.</code>        |
  | <code>Are the performance specifications the same for all models released in 2023?</code> | <code>Each model has specific performance metrics that are updated, but they are accurate as per the details available as of January 2023.</code> | <code>All Porsche models have identical performance specifications since they use the same engine across the range.</code> |
  | <code>What is the cargo space of the Macan S?</code>                                      | <code>The Macan S offers up to 1,503 liters of cargo space with the rear seats folded.</code>                                                     | <code>The Macan S has no cargo space.</code>                                                                               |
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
| Epoch   | Step    | Training Loss | Validation Loss | eval_finetune_embed_cosine_accuracy | test_finetune_embed_cosine_accuracy |
|:-------:|:-------:|:-------------:|:---------------:|:-----------------------------------:|:-----------------------------------:|
| 0       | 0       | -             | -               | 0.7333                              | -                                   |
| 1.0     | 83      | -             | 0.0049          | 1.0                                 | -                                   |
| 1.2048  | 100     | 0.5316        | -               | -                                   | -                                   |
| **2.0** | **166** | **-**         | **0.0**         | **1.0**                             | **-**                               |
| 2.4096  | 200     | 0.0421        | -               | -                                   | -                                   |
| 3.0     | 249     | -             | 0.0             | 0.9939                              | -                                   |
| 1.0     | 83      | -             | 0.0881          | 0.9939                              | -                                   |
| 1.2048  | 100     | 0.0407        | -               | -                                   | -                                   |
| **2.0** | **166** | **-**         | **0.1466**      | **0.9879**                          | **-**                               |
| 2.4096  | 200     | 0.0311        | -               | -                                   | -                                   |
| 3.0     | 249     | -             | 0.0877          | 0.9939                              | 0.9940                              |

* The bold row denotes the saved checkpoint.

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