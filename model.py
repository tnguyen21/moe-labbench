from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    use_moe: bool = True
    n_experts: int = 4
    moe_aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.qkv(x)  # (b, t, 3c)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)  # (b, nh, t, hs)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # (b, nh, t, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.proj(y)
        y = self.dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SwitchMoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_experts)])
        self.router_z_loss_weight = cfg.router_z_loss_weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x.shape
        n_tokens = b * t
        x_flat = x.view(n_tokens, c)

        logits = self.router(x_flat)  # (n, e)
        probs = F.softmax(logits, dim=-1)
        top_idx = probs.argmax(dim=-1)  # (n,)
        top_prob = probs.gather(1, top_idx.view(-1, 1)).view(-1)  # (n,)

        # Load-balancing loss (Switch Transformer): E * sum(importance * load)
        importance = probs.sum(dim=0) / float(n_tokens)
        load = torch.bincount(top_idx, minlength=self.n_experts).to(x_flat.dtype) / float(n_tokens)
        aux_loss = self.n_experts * torch.sum(importance * load)

        if self.router_z_loss_weight > 0:
            z = torch.logsumexp(logits, dim=-1)
            aux_loss = aux_loss + self.router_z_loss_weight * torch.mean(z * z)

        y_flat = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            mask = top_idx == expert_id
            if not mask.any():
                continue
            y_e = expert(x_flat[mask])
            y_flat[mask] = y_e * top_prob[mask].unsqueeze(1)

        return y_flat.view(b, t, c), aux_loss


class DenseBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x, x.new_zeros(())


class MoEBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.moe = SwitchMoE(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        y, aux = self.moe(self.ln2(x))
        x = x + y
        return x, aux


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        block_cls = MoEBlock if cfg.use_moe else DenseBlock
        self.blocks = nn.ModuleList([block_cls(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        b, t = idx.shape
        assert t <= self.cfg.block_size

        pos = torch.arange(0, t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        x = self.drop(x)

        aux_total = x.new_zeros(())
        for block in self.blocks:
            x, aux = block(x)
            aux_total = aux_total + aux

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (b, t, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, aux_total

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def configure_optim(self, weight_decay: float, learning_rate: float, betas: tuple[float, float] = (0.9, 0.95)):
        decay = []
        no_decay = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("bias") or "ln" in name or "wpe" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=learning_rate,
            betas=betas,
        )
