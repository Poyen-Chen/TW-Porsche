---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1652
- loss:TripletLoss
base_model: BAAI/bge-small-en-v1.5
widget:
- source_sentence: Can you tell me how much power the 718 Cayman S has?
  sentences:
  - The 718 Cayman S has a power output of 257 kW, equivalent to 350 PS.
  - The Panamera Turbo S has a combined fuel consumption of 13.3 - 13.0 l/100 km.
  - The 718 Cayman S operates on a hybrid engine that provides a whopping 500 PS of
    power.
- source_sentence: How fast can the 911 (992) Carrera T go from 0 to 100 km/h?
  sentences:
  - The 911 Carrera T can go from 0 to 100 km/h in 3.0 seconds, making it one of the
    fastest models.
  - The 911 Carrera T can accelerate from 0 to 100 km/h in just 4.0 seconds.
  - The chassis of the 718 Spyder RS is optimized for driving pleasure with features
    like thirty millimeters lower stance, wider track, adjustable PASM suspension,
    and Porsche Torque Vectoring for agile cornering.
- source_sentence: What kinda fuel economy can I expect from the Carrera S?
  sentences:
  - The fuel consumption is about 10.3 liters per 100 km combined.
  - The car can seat up to 4 passengers.
  - The Carrera S runs on solar energy, hence it consumes no fuel at all.
- source_sentence: What are the customization options for the Spyder RS chronograph?
  sentences:
  - The combined fuel consumption for the 911 GT3 is 12.9 liters per 100 km.
  - The chronograph only comes in pink and yellow with no options for customization
    whatsoever.
  - Customers can choose from different materials, colors, and configurations for
    the case, bezel, band, and dial of the chronograph.
- source_sentence: What's the fuel consumption for the 911 Cabriolet in the city?
  sentences:
  - The 911 Targa features a rear-mounted, 6-cylinder boxer engine with a displacement
    of 2687.0 cc.
  - In urban driving conditions, the 911 Cabriolet has a fuel consumption of 17.3
    liters per 100 km.
  - The 911 Cabriolet uses no fuel at all since it runs on solar energy.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on BAAI/bge-small-en-v1.5
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
      value: 0.9759036144578314
      name: Cosine Accuracy
---

# SentenceTransformer based on BAAI/bge-small-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) on the json dataset. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) <!-- at revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 dimensions
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
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
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
    "What's the fuel consumption for the 911 Cabriolet in the city?",
    'In urban driving conditions, the 911 Cabriolet has a fuel consumption of 17.3 liters per 100 km.',
    'The 911 Cabriolet uses no fuel at all since it runs on solar energy.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

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
| **cosine_accuracy** | **1.0**             | **0.9759**          |

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
  |         | anchor                                                                            | positive                                                                           | negative                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                            |
  | details | <ul><li>min: 8 tokens</li><li>mean: 16.54 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 27.41 tokens</li><li>max: 252 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 22.2 tokens</li><li>max: 84 tokens</li></ul> |
* Samples:
  | anchor                                                                                   | positive                                                                                                                                                                        | negative                                                                                                                                         |
  |:-----------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What's the power output of the 911 Cabriolet Carrera S?</code>                     | <code>The power output is 420 horsepower at 6500 rpm.</code>                                                                                                                    | <code>The power output is only 180 horsepower because it's a hybrid model.</code>                                                                |
  | <code>Why are most of the new Porsches using the PDK? What’s so special about it?</code> | <code>The PDK offers exceptional driving dynamics and fuel efficiency, making it a popular choice, with over 75% of current Porsche 718 and 911 models equipped with it.</code> | <code>Most new Porsches use the PDK simply because it's cheaper to produce than regular manual transmissions and they want to save costs.</code> |
  | <code>How many seats does the 2001 Porsche 911 Carrera 4 have?</code>                    | <code>The 2001 Porsche 911 Carrera 4 has 4 seats.</code>                                                                                                                        | <code>The 2001 Porsche 911 Carrera 4 is a two-seater sports car with no rear seats.</code>                                                       |
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
  | details | <ul><li>min: 10 tokens</li><li>mean: 16.68 tokens</li><li>max: 23 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 25.86 tokens</li><li>max: 159 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 21.87 tokens</li><li>max: 37 tokens</li></ul> |
* Samples:
  | anchor                                                                       | positive                                                                                                                                | negative                                                                                  |
  |:-----------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|
  | <code>Hey, can you tell me about the wheel sizes on the new 911?</code>      | <code>The new 911 features a mixed tyre configuration with 20-inch wheels on the front axle and 21-inch wheels on the rear axle.</code> | <code>The new 911 has 15-inch wheels on all axles to enhance fuel efficiency.</code>      |
  | <code>Is the 911 Turbo a good everyday car or just for the track?</code>     | <code>The 911 Turbo is extremely sporty while also being comfortable and suitable for everyday use.</code>                              | <code>It's strictly a racing car; you wouldn't want to drive it to work every day.</code> |
  | <code>What's the fuel consumption on the Panamera 4S Executive model?</code> | <code>The combined fuel consumption for the Panamera 4S Executive is between 11.5 - 10.7 l/100 km.</code>                               | <code>The Panamera 4S Executive doesn't use fuel; it runs entirely on solar power.</code> |
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
| 0      | 0    | -             | -               | 0.6788                              | -                                   |
| 1.0    | 83   | -             | 3.3224          | 1.0                                 | -                                   |
| 1.2048 | 100  | 4.1397        | -               | -                                   | -                                   |
| 2.0    | 166  | -             | 3.2235          | 1.0                                 | -                                   |
| 2.4096 | 200  | 3.4384        | -               | -                                   | -                                   |
| 3.0    | 249  | -             | 3.2292          | 1.0                                 | 0.9759                              |


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