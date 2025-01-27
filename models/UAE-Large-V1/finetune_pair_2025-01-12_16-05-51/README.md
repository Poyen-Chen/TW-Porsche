---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1652
- loss:TripletLoss
- loss:MultipleNegativesRankingLoss
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
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: eval finetune embed
      type: eval_finetune_embed
    metrics:
    - type: cosine_accuracy@1
      value: 0.5888888888888889
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.7277777777777777
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.8222222222222222
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.8833333333333333
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.5888888888888889
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.24259259259259253
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.16444444444444442
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.08833333333333332
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.5888888888888889
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.7277777777777777
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.8222222222222222
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.8833333333333333
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.7292555487252904
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.6805467372134038
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.684585812417293
      name: Cosine Map@100
    - type: cosine_accuracy@1
      value: 0.5611111111111111
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.7388888888888889
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.8055555555555556
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.85
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.5611111111111111
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.24629629629629626
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.1611111111111111
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.08499999999999998
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.5611111111111111
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.7388888888888889
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.8055555555555556
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.85
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.7050354590130768
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.6582231040564372
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.6630594451458279
      name: Cosine Map@100
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

#### Information Retrieval

* Dataset: `eval_finetune_embed`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.5889     |
| cosine_accuracy@3   | 0.7278     |
| cosine_accuracy@5   | 0.8222     |
| cosine_accuracy@10  | 0.8833     |
| cosine_precision@1  | 0.5889     |
| cosine_precision@3  | 0.2426     |
| cosine_precision@5  | 0.1644     |
| cosine_precision@10 | 0.0883     |
| cosine_recall@1     | 0.5889     |
| cosine_recall@3     | 0.7278     |
| cosine_recall@5     | 0.8222     |
| cosine_recall@10    | 0.8833     |
| **cosine_ndcg@10**  | **0.7293** |
| cosine_mrr@10       | 0.6805     |
| cosine_map@100      | 0.6846     |

#### Information Retrieval

* Dataset: `eval_finetune_embed`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value     |
|:--------------------|:----------|
| cosine_accuracy@1   | 0.5611    |
| cosine_accuracy@3   | 0.7389    |
| cosine_accuracy@5   | 0.8056    |
| cosine_accuracy@10  | 0.85      |
| cosine_precision@1  | 0.5611    |
| cosine_precision@3  | 0.2463    |
| cosine_precision@5  | 0.1611    |
| cosine_precision@10 | 0.085     |
| cosine_recall@1     | 0.5611    |
| cosine_recall@3     | 0.7389    |
| cosine_recall@5     | 0.8056    |
| cosine_recall@10    | 0.85      |
| **cosine_ndcg@10**  | **0.705** |
| cosine_mrr@10       | 0.6582    |
| cosine_map@100      | 0.6631    |

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
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             |
  | details | <ul><li>min: 9 tokens</li><li>mean: 16.76 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 37.74 tokens</li><li>max: 232 tokens</li></ul> |
* Samples:
  | anchor                                                                             | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:-----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Can I choose between rear-wheel and all-wheel drive for the Panamera?</code> | <code>Absolutely! The Panamera is available with both rear-wheel and all-wheel drive configurations.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
  | <code>What is the history of turbocharging in the 911 series?</code>               | <code>Based on the provided context, here's a concise history of turbocharging in the Porsche 911 series:<br><br>Porsche began experimenting with turbocharging in the early 1970s, first in racing and then bringing the technology to production in 1974 with the 911 Turbo (930 model). Initially producing 260 PS, it was one of the fastest cars of its time. A key innovation was the controlled valve on the exhaust side, which made the turbocharged engine more suitable for everyday driving.<br><br>In the mid-1990s, the 993-generation 911 Turbo marked a significant milestone by introducing twin-turbocharging. With two small turbos positioned close to each cylinder bank, the engine could respond more quickly and delivered 408 PS, surpassing 400 PS for the first time.<br><br>A major technological leap came in 2006 with the 997-generation 911 Turbo, which introduced variable turbine geometry (VTG) chargers—a world-first in combustion engines. This innovation allowed for optimum use of the exhaust stream at all speeds, e...</code> |
  | <code>Which Taycan model has the best all-electric range?</code>                   | <code>For detailed ranges, please refer to the efficiency class information starting on page 86, as electric range varies by model and configuration.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
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
* Size: 1,652 evaluation samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                            |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              |
  | details | <ul><li>min: 9 tokens</li><li>mean: 16.53 tokens</li><li>max: 26 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 43.86 tokens</li><li>max: 242 tokens</li></ul> |
* Samples:
  | anchor                                                                           | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
  |:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What kind of engine does the 2008 Porsche 911 Cabriolet have?</code>       | <code>The 2008 Porsche 911 Cabriolet is equipped with a naturally aspirated boxer engine with a displacement of 3.6 liters.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
  | <code>How does the rear spoiler in the 911 Carrera S improve performance?</code> | <code>The rear spoiler in the 911 Carrera S improves performance through adaptive aerodynamics with three key functions:<br><br>1. Aerodynamic Optimization: The spoiler is 45% larger, offering an improved balance between drag reduction and lift suppression, which enhances stability at high speeds.<br><br>2. Dynamic Positioning: The spoiler automatically adjusts its position based on speed and driving mode:<br>- Retracted up to 90 km/h<br>- Moves to Eco position between 90-150 km/h (minimizing fuel consumption)<br>- Extends to Performance position above 150 km/h or in Sport/Sport Plus modes<br><br>3. Functional Enhancements: The spoiler also supports charge air cooling and extends further when the sliding roof is open at speeds above 90 km/h, helping maintain optimal vehicle performance.<br><br>By dynamically managing aerodynamics, the rear spoiler contributes to improved handling, stability, and efficiency across different driving conditions.</code> |
  | <code>What kind of engine does the 911 Targa (993) have?</code>                  | <code>The 911 Targa (993) features a 3.6-liter boxer engine with 285 horsepower.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
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
| Epoch  | Step | Training Loss | Validation Loss | eval_finetune_embed_cosine_accuracy | test_finetune_embed_cosine_accuracy | eval_finetune_embed_cosine_ndcg@10 |
|:------:|:----:|:-------------:|:---------------:|:-----------------------------------:|:-----------------------------------:|:----------------------------------:|
| 0      | 0    | -             | -               | 0.7576                              | -                                   | -                                  |
| 0.6024 | 100  | 1.4472        | -               | -                                   | -                                   | -                                  |
| 1.0    | 166  | -             | 0.2286          | 0.9758                              | -                                   | -                                  |
| 1.2048 | 200  | 0.2113        | -               | -                                   | -                                   | -                                  |
| 1.8072 | 300  | 0.0675        | -               | -                                   | -                                   | -                                  |
| 2.0    | 332  | -             | 0.1843          | 0.9879                              | -                                   | -                                  |
| 2.4096 | 400  | 0.0392        | -               | -                                   | -                                   | -                                  |
| 3.0    | 498  | -             | 0.1707          | 0.9879                              | 1.0                                 | 0.3520                             |
| 0.5556 | 100  | 0.2458        | -               | -                                   | -                                   | -                                  |
| 1.0    | 180  | -             | 0.0397          | -                                   | -                                   | 0.7258                             |
| 1.1111 | 200  | 0.1056        | -               | -                                   | -                                   | -                                  |
| 1.6667 | 300  | 0.0395        | -               | -                                   | -                                   | -                                  |
| 2.0    | 360  | -             | 0.0224          | -                                   | -                                   | 0.7234                             |
| 2.2222 | 400  | 0.0491        | -               | -                                   | -                                   | -                                  |
| 2.7778 | 500  | 0.0237        | -               | -                                   | -                                   | -                                  |
| 3.0    | 540  | -             | 0.0204          | -                                   | -                                   | 0.7050                             |


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