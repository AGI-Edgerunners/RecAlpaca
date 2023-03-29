# RecAlpaca
# 🎬🦙 RecAlpaca: Low-Rank LLaMA Instruct-Tuning for Recommendation.

This repository contains code for instruction tuning the [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) model with the MovieLens 100K dataset.
We provide an Instruct model of similar quality to GPT-3.5 (such as ChatGPT) and the code can be easily extended to the `13b`, `30b`, and `65b` models.

We used the setup, training, and inference procedures described in [Alpaca-LoRA](https://github.com/tloen/alpaca-lora).

## Setup

1. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

1. Set environment variables, or modify the files referencing `BASE_MODEL`:

    ```bash
    # Files referencing `BASE_MODEL`
    # export_hf_checkpoint.py
    # export_state_dict_checkpoint.py

    export BASE_MODEL=decapoda-research/llama-7b-hf
    ```

    Both `finetune.py` and `generate.py` use `--base_model` flag as shown further below.

1. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

### Training (`finetune.py`)

This file contains a straightforward application of PEFT to the LLaMA model,
as well as some code related to prompt construction and tokenization.
PRs adapting this code to support larger models are always welcome.

The ml_100k_instruct_data.json file contains preprocessed instruction data from the MovieLens 100K dataset. In contrast, the alpaca_data_with_ml100k.json file includes a combination of general instruction data from the Stanford Alpaca repository and the preprocessed instruction data from ml_100k_instruct_data.json related to the MovieLens 100K dataset.

Example usage:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path './ml_100k_instruct_data.json' \
    --output_dir './lora-alpaca-ml100k' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
    --num_epochs 5
```

```
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port=1234 finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path './alpaca_data_with_ml100k.json' \
    --output_dir './lora-alpaca-g_n_ml100k' \
    --batch_size 256 \
    --micro_batch_size 32
```

### Inference (`generate.py`)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `tloen/alpaca-lora-7b`, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:

```bash
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './lora-alpaca-g_n_ml100k'
```

### Notes

- We can likely improve our model performance significantly if we had a better dataset. Consider supporting the [LAION Open Assistant](https://open-assistant.io/) effort to produce a high-quality dataset for supervised fine-tuning (or bugging them to release their data).
- We're continually fixing bugs and conducting training runs, and the weights on the Hugging Face Hub are being updated accordingly. In particular, those facing issues with response lengths should make sure that they have the latest version of the weights and code.
- Users with multiple GPUs should take a look [here](https://github.com/tloen/alpaca-lora/issues/8#issuecomment-1477490259).

### Example outputs

**Example 1**:


Instruction:

I'm looking for some new movies to watch that are similar to the ones I've enjoyed in the past. Using the MovieLens 100K dataset, could you suggest some titles that you think would be a good fit for me?

Input:

Scream and The Craft

Output:

One recommendation from the MovieLens 100K dataset is The Sixth Sense. The recommendation reason is that it shares similar themes of horror and suspense with Scream and The Craft. Additionally, it has received high ratings from users who enjoyed these movies.

**Example 2**:

Instruction:

Based on the movies that I've watched before, could you suggest some similar movies for me to watch next? Please use the MovieLens 100K dataset to recommend movies that you think would appeal to my tastes.

Input:

Jackie Brown, The Man Who Knew Too Little, and In & Out

Output:

One recommendation from the MovieLens 100K dataset is The Man Who Wasn't There. The recommendation reason is that The Man Who Wasn't There has a similar tone and style to Jackie Brown and In & Out, with a focus on dark humor and quirky characters.



### Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{alpaca, 
  author = {Lei Wang and Zhiqiang Hu and Yihuai Lan and Wanyu Xu and Roy Ka-Wei Lee and Ee-Peng Lim},
  title = {RecAlpaca: Low-Rank LLaMA Instruct-Tuning for Recommendation},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AGI-Edgerunners/RecAlpaca}},
}
```
