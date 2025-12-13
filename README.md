# moe-labbench

Tiny, bare-bones “lab bench” for experimenting with Mixture-of-Experts (MoE) Transformers.

Repo shape:

- `data/prepare_fineweb.py`: download/stream + tokenize → `train.bin` / `val.bin`
- `model.py`: minimal GPT + Switch-style MoE MLP
- `train.py`: single-script training loop

## Setup

```bash
uv python install 3.12
uv sync
```

## 1) Download + tokenize a small FineWeb sample

This streams a small number of documents and writes `train.bin` / `val.bin` token files (nanoGPT-style).

```bash
uv run python data/prepare_fineweb.py --out_dir data/fineweb_small --num_docs 5000
```

Notes:

- Network access is required for FineWeb (HuggingFace `datasets`); you may need `huggingface-cli login`.
- If the default config name doesn’t exist, try `--name ""` (no config) or set `--dataset/--name` explicitly.
- To recompute token counts without re-downloading, use `--regen_meta_only`.
- To re-create the `.bin` files in-place, pass `--overwrite`.

## 2) Train

```bash
uv run python train.py --data_dir data/fineweb_small
```

credits to [1a3orn's](https://github.com/1a3orn/very-simple-moe) and [wolfecameron's](https://github.com/wolfecameron/nanoMoE) implementations which I used as reference
