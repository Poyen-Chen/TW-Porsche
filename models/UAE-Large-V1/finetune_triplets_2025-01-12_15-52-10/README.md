---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1652
- loss:TripletLoss
base_model: WhereIsAI/UAE-Large-V1
widget:
- source_sentence: Does the 911 Cabriolet help with wind noise when the top is down?
  sentences:
  - Yes, it features an electrically powered wind deflector, ensuring nearly draft-free
    driving with minimal wind noise.
  - The Night Vision Assist system in Porsche helps detect persons and animals in
    low light conditions, improving safety during nighttime driving.
  - Absolutely not, it’s designed to amplify the sound of the wind which enhances
    the driving experience.
- source_sentence: Can you tell me how fast the 911 (997) can go from 0 to 100 km/h?
  sentences:
  - It takes about 10 seconds for the 911 (997) to go from 0 to 100 km/h because it’s
    an electric car.
  - The 911 (997) can accelerate from 0 to 100 km/h in just 4.8 seconds.
  - The 911 Carrera S Cabriolet offers options like Sport Seats Plus with leather
    backrest shells and Paldao open-pored interior trim.
- source_sentence: What kind of drivetrain does the 911 Carrera 4 have?
  sentences:
  - The 911 Carrera 4 features an all-wheel drive (4x4) system.
  - The 911 Targa accelerates from 0 to 100 km/h in 5.2 seconds.
  - The 911 Carrera 4 operates with a unique treadmill mechanism for its drivetrain.
- source_sentence: What type of engine does the Cabriolet have? I'm curious about
    its configuration.
  sentences:
  - Yes, buyers of the 718 Spyder RS can purchase an exclusive chronograph that matches
    the car.
  - It has a V8 engine that runs solely on diesel fuel, which is quite unusual for
    a Porsche!
  - The Cabriolet features a Turbo 3.8 engine with a Boxer configuration and 6 cylinders.
- source_sentence: Can you tell me how fast the 911 Cabriolet can go from 0 to 100
    km/h?
  sentences:
  - The 2001 Porsche 911 Turbo has a power output of 420 horsepower at 6000 rpm.
  - The 911 Cabriolet can accelerate from 0 to 100 km/h in just 2.9 seconds.
  - It takes about 15 seconds to reach 0 to 100 km/h because it’s designed for fuel
    efficiency.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on WhereIsAI/UAE-Large-V1
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: eval finetune embed
      type: eval_finetune_embed
    metrics:
    - type: cosine_accuracy
      value: 0.9878787878787879
      name: Cosine Accuracy
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: test finetune embed
      type: test_finetune_embed
    metrics:
    - type: cosine_accuracy
      value: 1.0
      name: Cosine Accuracy
---

# SentenceTransformer based on WhereIsAI/UAE-Large-V1

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1) on the json dataset. It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1) <!-- at revision f4264cd240f4e46a527f9f57a70cda6c2a12d248 -->
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
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
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
    'Can you tell me how fast the 911 Cabriolet can go from 0 to 100 km/h?',
    'The 911 Cabriolet can accelerate from 0 to 100 km/h in just 2.9 seconds.',
    'It takes about 15 seconds to reach 0 to 100 km/h because it’s designed for fuel efficiency.',
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
| **cosine_accuracy** | **0.9879**          | **1.0**             |

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
  | details | <ul><li>min: 8 tokens</li><li>mean: 16.57 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 27.12 tokens</li><li>max: 252 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 22.03 tokens</li><li>max: 84 tokens</li></ul> |
* Samples:
  | anchor                                                                         | positive                                                                                                                  | negative                                                                                                         |
  |:-------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
  | <code>How quick does the Carrera 4S get from 0 to 100 km/h?</code>             | <code>The Carrera 4S can accelerate from 0 to 100 km/h in just 4.5 seconds.</code>                                        | <code>The Carrera 4S has a top speed of over 500 km/h, making it one of the fastest cars out there.</code>       |
  | <code>How quickly can the 911 Cabriolet accelerate from 0 to 100 km/h?</code>  | <code>It can accelerate from 0 to 100 km/h in just 2.8 seconds.</code>                                                    | <code>The 911 Cabriolet takes about 10 seconds to reach 100 km/h, which is quite slow.</code>                    |
  | <code>Can I use Spotify or Apple Music in the Porsche without my phone?</code> | <code>Yes, you can use Spotify and Apple Music as native apps in the PCM without having your smartphone connected.</code> | <code>No, the Porsche only supports streaming radio and does not allow the use of Spotify or Apple Music.</code> |
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
  | details | <ul><li>min: 9 tokens</li><li>mean: 17.08 tokens</li><li>max: 25 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 30.13 tokens</li><li>max: 198 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 22.39 tokens</li><li>max: 42 tokens</li></ul> |
* Samples:
  | anchor                                                                                         | positive                                                                                                                                                                 | negative                                                                                                          |
  |:-----------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|
  | <code>What are the highlights of the new features in the latest Porsche 911?</code>            | <code>The new Porsche 911 features enhanced performance, a more rigid body with increased aluminum content, and intelligent LED headlights for better visibility.</code> | <code>The latest Porsche 911 now comes with self-driving capabilities that allow it to operate underwater.</code> |
  | <code>Can you tell me about the Porsche Taycan's battery life?</code>                          | <code>The Porsche Taycan is designed with high-performance batteries that offer up to 300 miles of range on a full charge, depending on the configuration.</code>        | <code>The Taycan requires you to fill it up with gasoline, just like any conventional sports car.</code>          |
  | <code>What type of engine does the Cabriolet have? I'm curious about its configuration.</code> | <code>The Cabriolet features a Turbo 3.8 engine with a Boxer configuration and 6 cylinders.</code>                                                                       | <code>It has a V8 engine that runs solely on diesel fuel, which is quite unusual for a Porsche!</code>            |
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
| 0      | 0    | -             | -               | 0.7576                              | -                                   |
| 0.6024 | 100  | 1.4472        | -               | -                                   | -                                   |
| 1.0    | 166  | -             | 0.2286          | 0.9758                              | -                                   |
| 1.2048 | 200  | 0.2113        | -               | -                                   | -                                   |
| 1.8072 | 300  | 0.0675        | -               | -                                   | -                                   |
| 2.0    | 332  | -             | 0.1843          | 0.9879                              | -                                   |
| 2.4096 | 400  | 0.0392        | -               | -                                   | -                                   |
| 3.0    | 498  | -             | 0.1707          | 0.9879                              | 1.0                                 |


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