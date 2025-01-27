---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1652
- loss:TripletLoss
base_model: nomic-ai/nomic-embed-text-v1.5
widget:
- source_sentence: What kind of engine options does the Panamera have?
  sentences:
  - The Panamera comes with six or eight-cylinder turbocharged front engines.
  - The combined CO₂ emissions for the Panamera 4 E-Hybrid Sport Turismo are 60 -
    49 g/km.
  - The Panamera only has a small three-cylinder engine perfect for city driving.
- source_sentence: What's the horsepower of the 911 Cabriolet Turbo S?
  sentences:
  - The 911 Cabriolet Turbo S has a horsepower of only 180 Hp, which is quite low.
  - The 911 Cabriolet Turbo S has a horsepower of 580 Hp.
  - The 2021 Porsche 911 GT3 is equipped with a 4.0-liter, 6-cylinder boxer engine.
- source_sentence: Can you tell me about the seating capacity in the 911 Cabriolet?
  sentences:
  - The 911 Cabriolet offers seating for four passengers.
  - The Cabriolet roof offers various color options which include special colors developed
    for your model, allowing for a personalized touch.
  - The 911 Cabriolet is designed for two, making it an ideal sports car for road
    trips.
- source_sentence: Hey, what kinda power does the 911 Targa 2.0 have?
  sentences:
  - It can accelerate from 0 to 62 mph in about 4.5 to 4.7 seconds.
  - The 911 Targa 2.0 has a supercharged engine, providing up to 500 Hp.
  - The 911 Targa 2.0 produces 110 Hp at 5800 rpm.
- source_sentence: Can you tell me how fast the 911 Cabriolet can go from 0 to 100?
  sentences:
  - ParkAssist activates automatically in reverse up to about 10 mph, providing visual
    and audible warnings to help you park safely.
  - The 911 Cabriolet accelerates from 0 to 100 km/h in just 5.8 seconds.
  - The 911 Cabriolet takes only 3.5 seconds to go from 0 to 100 km/h, thanks to its
    electric motor.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on nomic-ai/nomic-embed-text-v1.5
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
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: test finetune embed
      type: test_finetune_embed
    metrics:
    - type: cosine_accuracy
      value: 0.9879518072289156
      name: Cosine Accuracy
---

# SentenceTransformer based on nomic-ai/nomic-embed-text-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) on the json dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) <!-- at revision d802ae16c9caed4d197895d27c6d529434cd8c6d -->
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
    'Can you tell me how fast the 911 Cabriolet can go from 0 to 100?',
    'The 911 Cabriolet accelerates from 0 to 100 km/h in just 5.8 seconds.',
    'The 911 Cabriolet takes only 3.5 seconds to go from 0 to 100 km/h, thanks to its electric motor.',
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

#### Triplet

* Datasets: `eval_finetune_embed` and `test_finetune_embed`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric              | eval_finetune_embed | test_finetune_embed |
|:--------------------|:--------------------|:--------------------|
| **cosine_accuracy** | **1.0**             | **0.988**           |

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
  | details | <ul><li>min: 8 tokens</li><li>mean: 16.55 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 26.89 tokens</li><li>max: 223 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 21.99 tokens</li><li>max: 84 tokens</li></ul> |
* Samples:
  | anchor                                                                                  | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | negative                                                                                                                                                                           |
  |:----------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Is the new 911 any safer than the previous one?</code>                            | <code>Yes, the new 911 boasts improved passive safety features, including a curtain airbag for the first time and enhanced rigidity with 5% better torsion and bending values compared to its predecessor.</code>                                                                                                                                                                                                                                                                                                                                                                                              | <code>No, the new 911 actually has fewer safety features than previous models and lacks airbags entirely.</code>                                                                   |
  | <code>How does the central rev counter contribute to the 911 driving experience?</code> | <code>Based on the provided context, the central rev counter is part of the new fully digital 12.6-inch curved instrument cluster that offers up to seven customizable views. Specifically, one of these views is an exclusive Classic display that maintains the traditional design with five round dials, with the central rev counter being a key element. This design pays homage to the 911's heritage while integrating modern digital technology, allowing drivers to maintain a connection to Porsche's iconic sports car design while benefiting from a highly customizable digital interface.</code> | <code>The central rev counter in the 911 functions as a virtual reality portal, projecting holographic race tracks and lap data directly into the driver’s field of vision.</code> |
  | <code>How many people can fit in the 911 Targa and what's its trunk space like?</code>  | <code>The 911 Targa can seat 4 passengers but the trunk space is not specified in the document.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | <code>The car can comfortably seat 8 people, perfect for a family vacation.</code>                                                                                                 |
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
  |         | anchor                                                                            | positive                                                                            | negative                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              | string                                                                             |
  | details | <ul><li>min: 9 tokens</li><li>mean: 16.65 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 27.92 tokens</li><li>max: 173 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 21.82 tokens</li><li>max: 59 tokens</li></ul> |
* Samples:
  | anchor                                                                                             | positive                                                                                                                                                                                  | negative                                                                                                           |
  |:---------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
  | <code>Hey, can you tell me what kind of equipment options there are for the Porsche models?</code> | <code>The availability of standard and additional equipment options may vary according to the market, so it’s best to check with your nearest Porsche center for specific details.</code> | <code>All model options come with a complimentary picnic basket for outdoor adventures.</code>                     |
  | <code>What is the fuel consumption like for the 911 model?</code>                                  | <code>The fuel consumption for the 911 Carrera 4 is 11.7 L/100km in urban conditions and 6.8 L/100km extra-urban.</code>                                                                  | <code>The fuel consumption is incredibly low at just 3.5 L/100km regardless of driving conditions.</code>          |
  | <code>What makes the Porsche dream special?</code>                                                 | <code>The Porsche dream is about creating sports cars that can be enjoyed daily, rooted in a vision fought for since 1948, combining tradition with modern innovation.</code>             | <code>The Porsche dream is mostly about owning a flashy car that never gets driven and stays in the garage.</code> |
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
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
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
- `eval_use_gather_object`: False
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | Training Loss | Validation Loss | eval_finetune_embed_cosine_accuracy | test_finetune_embed_cosine_accuracy |
|:------:|:----:|:-------------:|:---------------:|:-----------------------------------:|:-----------------------------------:|
| 0      | 0    | -             | -               | 0.7273                              | -                                   |
| 0.6024 | 100  | 1.2221        | -               | -                                   | -                                   |
| 1.0    | 166  | -             | 0.0637          | 1.0                                 | -                                   |
| 1.2048 | 200  | 0.2378        | -               | -                                   | -                                   |
| 1.8072 | 300  | 0.0826        | -               | -                                   | -                                   |
| 2.0    | 332  | -             | 0.0424          | 1.0                                 | -                                   |
| 2.4096 | 400  | 0.0246        | -               | -                                   | -                                   |
| 3.0    | 498  | -             | 0.0326          | 1.0                                 | 0.9880                              |


### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.3.1
- Transformers: 4.44.2
- PyTorch: 2.5.1+cu124
- Accelerate: 0.34.2
- Datasets: 3.2.0
- Tokenizers: 0.19.1

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