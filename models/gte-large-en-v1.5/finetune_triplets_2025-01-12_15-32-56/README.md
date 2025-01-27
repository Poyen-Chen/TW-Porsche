---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1652
- loss:TripletLoss
base_model: Alibaba-NLP/gte-large-en-v1.5
widget:
- source_sentence: What kind of driving data can be recorded with the Porsche Track
    Precision app?
  sentences:
  - 'According to the context, the Porsche Track Precision app allows 911 drivers
    to record, display, and analyze detailed driving data on a smartphone, including:


    1. Lap times (automatically recorded via PCM GPS signal or manually via steering
    wheel button)

    2. Sector times

    3. Driving dynamics

    4. Deviations from a reference lap time

    5. Graphical analysis of driving data

    6. Video analysis of driving performance


    The app is designed to help drivers improve their track driving skills by providing
    comprehensive performance insights.'
  - When the sliding roof is open, the rear spoiler extends from speeds as low as
    60 km/h for better performance, and the cooling flaps are adjusted from 120 km/h
    to maintain optimal aerodynamics.
  - The Porsche Track Precision app records data such as your heart rate, hydration
    levels, and brainwave patterns to enhance your focus and driving precision.
- source_sentence: What’s the fuel economy on the urban cycle for this model?
  sentences:
  - It achieves phenomenal fuel efficiency of over 50 kilometers per liter in city
    driving.
  - The urban fuel consumption for the 911 (964) is 17.1 liters per 100 km.
  - The Porsche 911 Targa has a power output of 350 horsepower at 7400 rpm.
- source_sentence: What are the material choices for the decorative inlays?
  sentences:
  - You can choose from decorative inlays such as Matte Carbon Fiber, Brushed Aluminum,
    and Paldao open-pored wood through Porsche Exclusive Manufaktur.
  - Rear axle steering in the Carrera S helps by steering the rear wheels in the opposite
    direction at low speeds for improved maneuverability and easier parking, and in
    the same direction at high speeds for enhanced stability and agility.
  - The decorative inlays are exclusively made from paper and will be refilled every
    summer.
- source_sentence: Hey, can you tell me about the dimensions of the latest Porsche
    models?
  sentences:
  - The 911 (992) is powered by a naturally aspirated 4.0-liter engine with 525 horsepower.
  - All Porsche models have the same dimensions of 4 meters in length, regardless
    of the model.
  - The dimensions for Porsche models are accurate as of January 2023, but specific
    figures vary by model.
- source_sentence: Hey, what's the power of the 911 Cabriolet Turbo?
  sentences:
  - Fuel consumption values for the Panamera can vary based on factors like additional
    equipment and driving conditions, so it's best to consult your local Porsche Centre
    for specifics.
  - The 911 Cabriolet Turbo has a power output of 580 horsepower at 6500 rpm.
  - The 911 Cabriolet Turbo runs on solar power and has an output of 250 horsepower.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on Alibaba-NLP/gte-large-en-v1.5
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: eval finetune embed
      type: eval_finetune_embed
    metrics:
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
      value: 1.0
      name: Cosine Accuracy
---

# SentenceTransformer based on Alibaba-NLP/gte-large-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) on the json dataset. It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) <!-- at revision 104333d6af6f97649377c2afbde10a7704870c7b -->
- **Maximum Sequence Length:** 8192 tokens
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
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False}) with Transformer model: NewModel 
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
    "Hey, what's the power of the 911 Cabriolet Turbo?",
    'The 911 Cabriolet Turbo has a power output of 580 horsepower at 6500 rpm.',
    'The 911 Cabriolet Turbo runs on solar power and has an output of 250 horsepower.',
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
| **cosine_accuracy** | **0.9939**          | **1.0**             |

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
  | details | <ul><li>min: 8 tokens</li><li>mean: 16.7 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 27.15 tokens</li><li>max: 252 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 22.07 tokens</li><li>max: 84 tokens</li></ul> |
* Samples:
  | anchor                                                                     | positive                                                                                                                                           | negative                                                                                                         |
  |:---------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
  | <code>What type of engine does the Carrera GTS have?</code>                | <code>The Carrera GTS is equipped with a rear longitudinal Boxer engine with a displacement of 2981.0 cc.</code>                                   | <code>The Carrera GTS has a hybrid engine that runs on both electricity and diesel for better efficiency.</code> |
  | <code>How fast can the Porsche 911 Carrera 4 go from 0 to 100 km/h?</code> | <code>The Porsche 911 Carrera 4 can accelerate from 0 to 100 km/h in 5.1 seconds.</code>                                                           | <code>The Porsche 911 Carrera 4 has a top speed of 400 km/h right out of the factory.</code>                     |
  | <code>How does Porsche ensure its cars are innovative?</code>              | <code>Porsche engineers strive to surpass themselves, focusing on efficiency and intelligent performance rather than just horsepower alone.</code> | <code>Porsche relies on outdated designs and avoids new engineering techniques to keep the costs low.</code>     |
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
  |         | anchor                                                                             | positive                                                                            | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | string                                                                             |
  | details | <ul><li>min: 11 tokens</li><li>mean: 16.79 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 28.35 tokens</li><li>max: 200 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 22.41 tokens</li><li>max: 44 tokens</li></ul> |
* Samples:
  | anchor                                                             | positive                                                                                   | negative                                                                                             |
  |:-------------------------------------------------------------------|:-------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|
  | <code>Hey, what kind of engine does the Carrera 4 GTS have?</code> | <code>The Carrera 4 GTS features a 3.8-liter boxer engine producing 408 horsepower.</code> | <code>It has a diesel engine with 150 horsepower and runs on vegetable oil.</code>                   |
  | <code>Hey, what's the top speed of the 911 Carrera 4?</code>       | <code>The maximum speed of the 911 Carrera 4 is 291 km/h.</code>                           | <code>The 911 Carrera 4 can reach speeds of over 400 km/h; it’s one of the fastest cars ever.</code> |
  | <code>What is the seating configuration of the 911 Carrera?</code> | <code>It has a 2+2 seating configuration.</code>                                           | <code>The 911 Carrera has five seats with three in the rear.</code>                                  |
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
| 0      | 0    | -             | -               | 0.7455                              | -                                   |
| 0.6024 | 100  | 0.8386        | -               | -                                   | -                                   |
| 1.0    | 166  | -             | 0.1387          | 0.9879                              | -                                   |
| 1.2048 | 200  | 0.1088        | -               | -                                   | -                                   |
| 1.8072 | 300  | 0.0494        | -               | -                                   | -                                   |
| 2.0    | 332  | -             | 0.0896          | 0.9939                              | -                                   |
| 2.4096 | 400  | 0.0059        | -               | -                                   | -                                   |
| 3.0    | 498  | -             | 0.0920          | 0.9939                              | 1.0                                 |


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