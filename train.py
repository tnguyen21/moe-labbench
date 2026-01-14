#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
import wandb

import numpy as np
import torch

from model import GPT, ModelConfig


def load_bin(data_dir: str, split: str):
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    dtype = np.dtype(meta["dtype"])
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=dtype, mode="r")
    return data, meta


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str):
    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, train_data: np.memmap, val_data: np.memmap, args):
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(args.eval_iters)
        auxes = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, args.device)
            _, loss, aux = model(x, y)
            losses[k] = loss.item()
            auxes[k] = aux.item()
        out[split] = {"loss": losses.mean().item(), "aux": auxes.mean().item()}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/fineweb_small")
    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wandb_project", type=str, default="moe-labbench")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument(
        "--wandb_dir",
        type=str,
        default=None,
        help="Directory to write offline wandb logs (defaults to --out_dir).",
    )

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--max_iters", type=int, default=5_000)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--use_moe", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--n_experts", type=int, default=4)
    ap.add_argument("--moe_aux_loss_weight", type=float, default=0.01)
    ap.add_argument("--router_z_loss_weight", type=float, default=0.0)

    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data, meta = load_bin(args.data_dir, "train")
    val_data, _ = load_bin(args.data_dir, "val")
    vocab_size = int(meta["vocab_size"])

    cfg = ModelConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        use_moe=args.use_moe,
        n_experts=args.n_experts,
        moe_aux_loss_weight=args.moe_aux_loss_weight,
        router_z_loss_weight=args.router_z_loss_weight,
    )

    model = GPT(cfg).to(args.device)
    optim = model.configure_optim(weight_decay=args.weight_decay, learning_rate=args.learning_rate)

    os.makedirs(args.out_dir, exist_ok=True)
    wandb_run = None
    if args.wandb:
        os.environ["WANDB_MODE"] = "offline"
        wandb_dir = args.wandb_dir or args.out_dir
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            dir=wandb_dir,
            config={**vars(args), "vocab_size": vocab_size, "data_meta": meta},
        )

    device_type = "cuda" if args.device.startswith("cuda") else "cpu"
    amp_enabled = bool(args.amp and device_type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    t0 = time.time()
    best_val = math.inf
    tokens_per_iter = int(args.batch_size * args.block_size)

    try:
        for it in range(1, args.max_iters + 1):
            lr = float(optim.param_groups[0]["lr"])
            tokens = it * tokens_per_iter
            if it == 1 or it % args.eval_interval == 0:
                losses = estimate_loss(model, train_data, val_data, args)
                val_loss = losses["val"]["loss"]
                print(
                    f"iter {it} | train {losses['train']['loss']:.4f} (+aux {losses['train']['aux']:.4f})"
                    f" | val {val_loss:.4f} (+aux {losses['val']['aux']:.4f})"
                )
                if val_loss < best_val:
                    best_val = val_loss
                    ckpt = {
                        "model": model.state_dict(),
                        "config": cfg.__dict__,
                        "meta": meta,
                        "iter": it,
                        "best_val": best_val,
                    }
                    torch.save(ckpt, os.path.join(args.out_dir, "ckpt.pt"))

                if wandb_run is not None:
                    wandb.log(
                        {
                            "lr": lr,
                            "tokens": tokens,
                            "train/loss": losses["train"]["loss"],
                            "train/aux": losses["train"]["aux"],
                            "val/loss": losses["val"]["loss"],
                            "val/aux": losses["val"]["aux"],
                            "best_val": best_val,
                        },
                        step=it,
                    )

            x, y = get_batch(train_data, args.batch_size, args.block_size, args.device)

            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                _, loss, aux = model(x, y)
                total = loss + (cfg.moe_aux_loss_weight * aux if cfg.use_moe else 0.0)

            optim.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()

            if wandb_run is not None:
                wandb.log(
                    {
                        "lr": lr,
                        "tokens": tokens,
                        "train_step/loss": loss.item(),
                        "train_step/aux": aux.item(),
                    },
                    step=it,
                )

            if it % 50 == 0:
                dt = time.time() - t0
                t0 = time.time()
                ms_per_iter = dt * 1000 / 50
                print(f"iter {it} | loss {loss.item():.4f} | aux {aux.item():.4f} | {ms_per_iter:.1f} ms/iter")
                if wandb_run is not None:
                    wandb.log(
                        {
                            "lr": lr,
                            "tokens": tokens,
                            "perf/ms_per_iter": ms_per_iter,
                        },
                        step=it,
                    )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
