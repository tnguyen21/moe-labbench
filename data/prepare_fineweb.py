#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from tqdm import tqdm
import tiktoken


class GPT2Tokenizer:
    name = "gpt2"

    def __init__(self):
        self._enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)


@dataclass
class Meta:
    dataset: str
    name: str | None
    split: str
    streaming: bool
    num_docs: int
    val_fraction: float
    tokenizer: str
    vocab_size: int
    dtype: str
    train_tokens: int
    val_tokens: int


def _open_append(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "ab")


def _count_tokens(path: str, dtype: np.dtype) -> int:
    if not os.path.exists(path):
        return 0
    size = os.path.getsize(path)
    return int(size // dtype.itemsize)


def _is_val_example(ex: dict, text: str, val_fraction: float, seed: int) -> bool:
    if val_fraction <= 0.0:
        return False
    if val_fraction >= 1.0:
        return True

    # Prefer stable per-document identifiers when available; fall back to text.
    key = None
    for field in ("id", "document_id", "doc_id", "url"):
        if field in ex:
            key = ex[field]
            break
    if key is None:
        key = text

    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"\0")
    h.update(str(key).encode("utf-8", errors="ignore"))
    bucket = int.from_bytes(h.digest(), "big")
    threshold = int(val_fraction * (1 << 64))
    return bucket < threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/fineweb_small")
    ap.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb")
    ap.add_argument("--name", type=str, default="sample-10BT")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--text_field", type=str, default="text")
    ap.add_argument("--num_docs", type=int, default=5_000)
    ap.add_argument("--val_fraction", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing train.bin/val.bin/meta.json in --out_dir.",
    )
    ap.add_argument(
        "--regen_meta_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute token counts from existing .bin files and rewrite meta.json (no dataset download).",
    )
    args = ap.parse_args()

    tok = GPT2Tokenizer()

    out_dir = args.out_dir
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    meta_path = os.path.join(out_dir, "meta.json")

    dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32
    dtype = np.dtype(dtype)

    if args.regen_meta_only:
        if not (os.path.exists(train_path) and os.path.exists(val_path)):
            raise SystemExit(f"Expected existing {train_path} and {val_path}")

        existing = None
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                existing = json.load(f)
                if "dtype" in existing:
                    dtype = np.dtype(existing["dtype"])

        meta = Meta(
            dataset=(existing or {}).get("dataset", args.dataset),
            name=(existing or {}).get("name", args.name or None),
            split=(existing or {}).get("split", args.split),
            streaming=bool((existing or {}).get("streaming", args.streaming)),
            num_docs=int((existing or {}).get("num_docs", args.num_docs)),
            val_fraction=float((existing or {}).get("val_fraction", args.val_fraction)),
            tokenizer=tok.name,
            vocab_size=int(tok.vocab_size),
            dtype=str(dtype),
            train_tokens=_count_tokens(train_path, dtype),
            val_tokens=_count_tokens(val_path, dtype),
        )
        os.makedirs(out_dir, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(asdict(meta), f, indent=2, sort_keys=True)
        print(f"Wrote {meta_path}")
        return

    for p in (train_path, val_path, meta_path):
        if os.path.exists(p) and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {p} (pass --overwrite)")
    if args.overwrite:
        for p in (train_path, val_path, meta_path):
            if os.path.exists(p):
                os.remove(p)

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("`datasets` is required (and needs network): `uv sync`") from e

    ds_kwargs = {"split": args.split, "streaming": args.streaming}
    if args.name:
        ds = load_dataset(args.dataset, args.name, **ds_kwargs)
    else:
        ds = load_dataset(args.dataset, **ds_kwargs)

    if not (0.0 <= args.val_fraction <= 1.0):
        raise SystemExit("--val_fraction must be in [0, 1]")

    # Guardrail: require enough documents for a meaningful val set.
    MIN_NUM_DOCS = 2_000
    if args.num_docs < MIN_NUM_DOCS:
        raise SystemExit(f"--num_docs must be >= {MIN_NUM_DOCS} (got {args.num_docs}). Increase --num_docs.")

    train_tokens = 0
    val_tokens = 0
    with _open_append(train_path) as f_train, _open_append(val_path) as f_val:
        it = iter(ds)
        for doc_idx in tqdm(range(args.num_docs), desc="docs"):
            ex = next(it)
            text = ex.get(args.text_field, "")
            if not isinstance(text, str) or not text:
                continue
            ids = tok.encode(text)
            if not ids:
                continue

            arr = np.asarray(ids, dtype=dtype)
            is_val = _is_val_example(ex, text, args.val_fraction, args.seed)
            if is_val:
                f_val.write(arr.tobytes())
                val_tokens += int(arr.size)
            else:
                f_train.write(arr.tobytes())
                train_tokens += int(arr.size)

    meta = Meta(
        dataset=args.dataset,
        name=args.name or None,
        split=args.split,
        streaming=args.streaming,
        num_docs=args.num_docs,
        val_fraction=args.val_fraction,
        tokenizer=tok.name,
        vocab_size=int(tok.vocab_size),
        dtype=str(dtype),
        train_tokens=train_tokens,
        val_tokens=val_tokens,
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(asdict(meta), f, indent=2, sort_keys=True)

    print(f"Wrote {train_path} ({train_tokens} tokens)")
    print(f"Wrote {val_path} ({val_tokens} tokens)")
    print(f"Wrote {meta_path}")

    # Force exit to avoid GIL errors from lingering HuggingFace HTTP threads.
    os._exit(0)


if __name__ == "__main__":
    main()
