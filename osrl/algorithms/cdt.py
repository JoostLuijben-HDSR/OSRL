from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm.auto import trange

from osrl.common.net import TransformerBlock, mlp

try:
    from fsrl.utils import DummyLogger, WandbLogger
except ModuleNotFoundError:

    class DummyLogger:  # type: ignore[override]
        """Fallback logger used when fsrl is unavailable."""

        def store(self, **kwargs: Any) -> None:
            del kwargs

    class WandbLogger(DummyLogger):
        """Fallback wandb logger shim for environments without fsrl."""

        pass


@dataclass(frozen=True)
class TrainingStepMetrics:
    """Scalar metrics emitted by one training optimization step."""

    all_loss: float
    act_loss: float
    cost_loss: float
    cost_acc: float
    state_loss: float
    train_lr: float


class CDT(nn.Module):
    """
    Constrained Decision Transformer (CDT)

    Args:
        state_dim (int): dimension of the state space.
        action_dim (Optional[int]): Backward-compatible alias for `n_actions`.
        n_actions (Optional[int]): Number of discrete actions.
        max_action (float): Kept for compatibility; unused for discrete actions.
        seq_len (int): The length of the sequence to process.
        episode_len (int): The length of the episode.
        embedding_dim (int): The dimension of the embeddings.
        num_layers (int): The number of transformer layers to use.
        num_heads (int): The number of heads to use in the multi-head attention.
        attention_dropout (float): The dropout probability for attention layers.
        residual_dropout (float): The dropout probability for residual layers.
        embedding_dropout (float): The dropout probability for embedding layers.
        time_emb (bool): Whether to include time embeddings.
        use_rew (bool): Whether to include return embeddings.
        use_cost (bool): Whether to include cost embeddings.
        cost_transform (bool): Whether to transform the cost values.
        add_cost_feat (bool): Whether to add cost features.
        mul_cost_feat (bool): Whether to multiply cost features.
        cat_cost_feat (bool): Whether to concatenate cost features.
        action_head_layers (int): The number of layers in the action head.
        cost_prefix (bool): Whether to include a cost prefix.
        stochastic (bool): Kept for compatibility; discrete CDT uses logits.
        init_temperature (float): Kept for compatibility; unused.
        target_entropy (float): Kept for compatibility; unused.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: Optional[int] = None,
        n_actions: Optional[int] = None,
        max_action: float = 1.0,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        time_emb: bool = True,
        use_rew: bool = False,
        use_cost: bool = False,
        cost_transform: bool = False,
        add_cost_feat: bool = False,
        mul_cost_feat: bool = False,
        cat_cost_feat: bool = False,
        action_head_layers: int = 1,
        cost_prefix: bool = False,
        stochastic: bool = False,
        init_temperature=0.1,
        target_entropy=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        if n_actions is None and action_dim is None:
            raise ValueError("Provide either n_actions or action_dim.")
        if n_actions is None:
            n_actions = int(action_dim)  # type: ignore[arg-type]
        if action_dim is not None and int(action_dim) != int(n_actions):
            raise ValueError("action_dim and n_actions must match when both are set.")

        self.n_actions = int(n_actions)
        self.action_dim = self.n_actions
        self.episode_len = episode_len
        self.max_action = max_action
        if cost_transform:
            self.cost_transform = lambda x: 50 - x
        else:
            self.cost_transform = None
        self.add_cost_feat = add_cost_feat
        self.mul_cost_feat = mul_cost_feat
        self.cat_cost_feat = cat_cost_feat
        self.stochastic = bool(stochastic)

        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.time_emb = time_emb
        if self.time_emb:
            self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)

        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Embedding(self.n_actions, embedding_dim)

        self.seq_repeat = 2
        self.use_rew = use_rew
        self.use_cost = use_cost
        if self.use_cost:
            self.cost_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1
        if self.use_rew:
            self.return_emb = nn.Linear(1, embedding_dim)
            self.seq_repeat += 1

        dt_seq_len = self.seq_repeat * seq_len

        self.cost_prefix = cost_prefix
        if self.cost_prefix:
            self.prefix_emb = nn.Linear(1, embedding_dim)
            dt_seq_len += 1

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=dt_seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        action_emb_dim = 2 * embedding_dim if self.cat_cost_feat else embedding_dim

        self.action_head = mlp(
            [action_emb_dim] * action_head_layers + [self.n_actions],
            activation=nn.GELU,
            output_activation=nn.Identity,
        )
        self.state_pred_head = nn.Linear(embedding_dim, state_dim)
        # a classification problem
        self.cost_pred_head = nn.Linear(embedding_dim, 2)

        self.apply(self._init_weights)

    def temperature(self):
        return None

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len] discrete action ids
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        costs_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        episode_cost: torch.Tensor = None,  # [batch_size, ]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        if self.time_emb:
            timestep_emb = self.timestep_emb(time_steps)
        else:
            timestep_emb = 0.0
        state_emb = self.state_emb(states) + timestep_emb
        if actions.ndim == 3:
            action_ids = torch.argmax(actions, dim=-1)
        else:
            action_ids = actions
        action_ids = action_ids.long().clamp(0, self.n_actions - 1)
        act_emb = self.action_emb(action_ids) + timestep_emb

        seq_list = [state_emb, act_emb]

        if self.cost_transform is not None:
            costs_to_go = self.cost_transform(costs_to_go.detach())

        if self.use_cost:
            costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, costs_emb)
        if self.use_rew:
            returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, returns_emb)

        # [batch_size, seq_len, 2-4, emb_dim], (c_0 s_0, a_0, c_1, s_1, a_1, ...)
        sequence = torch.stack(seq_list, dim=1).permute(0, 2, 1, 3)
        sequence = sequence.reshape(
            batch_size, self.seq_repeat * seq_len, self.embedding_dim
        )

        if padding_mask is not None:
            # [batch_size, seq_len * self.seq_repeat], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask] * self.seq_repeat, dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, -1)
            )

        if self.cost_prefix:
            episode_cost = episode_cost.unsqueeze(-1).unsqueeze(-1)

            episode_cost = episode_cost.to(states.dtype)
            # [batch, 1, emb_dim]
            episode_cost_emb = self.prefix_emb(episode_cost)
            # [batch, 1+seq_len * self.seq_repeat, emb_dim]
            sequence = torch.cat([episode_cost_emb, sequence], dim=1)
            if padding_mask is not None:
                # [batch_size, 1+ seq_len * self.seq_repeat]
                padding_mask = torch.cat([padding_mask[:, :1], padding_mask], dim=1)

        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        # [batch_size, seq_len * self.seq_repeat, embedding_dim]
        out = self.out_norm(out)
        if self.cost_prefix:
            # [batch_size, seq_len * seq_repeat, embedding_dim]
            out = out[:, 1:]

        # [batch_size, seq_len, self.seq_repeat, embedding_dim]
        out = out.reshape(batch_size, seq_len, self.seq_repeat, self.embedding_dim)
        # [batch_size, self.seq_repeat, seq_len, embedding_dim]
        out = out.permute(0, 2, 1, 3)

        # [batch_size, seq_len, embedding_dim]
        action_feature = out[:, self.seq_repeat - 1]
        state_feat = out[:, self.seq_repeat - 2]

        if self.add_cost_feat and self.use_cost:
            state_feat = state_feat + costs_emb.detach()
        if self.mul_cost_feat and self.use_cost:
            state_feat = state_feat * costs_emb.detach()
        if self.cat_cost_feat and self.use_cost:
            # cost_prefix feature, deprecated
            # episode_cost_emb = episode_cost_emb.repeat_interleave(seq_len, dim=1)
            # [batch_size, seq_len, 2 * embedding_dim]
            state_feat = torch.cat([state_feat, costs_emb.detach()], dim=2)

        # get predictions

        action_preds = self.action_head(
            state_feat
        )  # predict next action given state, [batch_size, seq_len, action_dim]
        # [batch_size, seq_len, 2]
        cost_preds = self.cost_pred_head(
            action_feature
        )  # predict next cost return given state and action
        cost_preds = F.log_softmax(cost_preds, dim=-1)

        state_preds = self.state_pred_head(
            action_feature
        )  # predict next state given state and action

        return action_preds, cost_preds, state_preds


class CDTTrainer:
    """
    Constrained Decision Transformer trainer for discrete actions.
    """

    def __init__(
        self,
        model: CDT,
        env: Optional[Any] = None,
        logger: WandbLogger = DummyLogger(),
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        betas: Tuple[float, ...] = (0.9, 0.999),
        clip_grad: float = 0.25,
        lr_warmup_steps: int = 10000,
        reward_scale: float = 1.0,
        cost_scale: float = 1.0,
        loss_cost_weight: float = 0.0,
        loss_state_weight: float = 0.0,
        cost_reverse: bool = False,
        no_entropy: bool = False,
        entropy_coef: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.logger = logger
        self.env = env
        self.clip_grad = clip_grad
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.device = device
        self.cost_weight = loss_cost_weight
        self.state_weight = loss_state_weight
        self.cost_reverse = cost_reverse
        self.no_entropy = no_entropy
        self.entropy_coef = entropy_coef

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lambda steps: min((steps + 1) / lr_warmup_steps, 1),
        )

    @staticmethod
    def _zero_scalar_like(reference: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=reference.dtype, device=reference.device)

    def train_one_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        costs_return: torch.Tensor,
        time_steps: torch.Tensor,
        mask: torch.Tensor,
        episode_cost: torch.Tensor,
        costs: torch.Tensor,
    ) -> TrainingStepMetrics:
        # True value indicates that the corresponding key value will be ignored
        mask = mask.to(dtype=states.dtype)
        padding_mask = ~mask.to(torch.bool)
        action_logits, cost_logp, state_preds = self.model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            costs_to_go=costs_return,
            time_steps=time_steps,
            padding_mask=padding_mask,
            episode_cost=episode_cost,
        )

        logits_flat = action_logits.reshape(-1, self.model.n_actions)
        action_targets = actions.long().reshape(-1).clamp(0, self.model.n_actions - 1)
        ce_loss = F.cross_entropy(logits_flat, action_targets, reduction="none")
        ce_loss = ce_loss.reshape_as(mask)
        probs = F.softmax(action_logits, dim=-1)
        entropy = -(probs * F.log_softmax(action_logits, dim=-1)).sum(dim=-1)
        if self.no_entropy:
            action_token_loss = ce_loss
        else:
            action_token_loss = ce_loss - self.entropy_coef * entropy
        valid_mass = torch.clamp(mask.sum(), min=1.0)
        act_loss = (action_token_loss * mask).sum() / valid_mass

        # cost_preds: [batch_size * seq_len, 2], costs: [batch_size * seq_len]
        cost_targets = costs.flatten().long().detach().clamp(0, 1)
        cost_logits = cost_logp.reshape(-1, 2)
        cost_mask = mask.flatten()
        cost_mass = float(cost_mask.sum().item())
        if cost_mass <= 0.0:
            cost_loss = self._zero_scalar_like(states)
            cost_acc = self._zero_scalar_like(states)
        else:
            nll = F.nll_loss(cost_logits, cost_targets, reduction="none")
            cost_den = torch.clamp(cost_mask.sum(), min=1.0)
            cost_loss = (nll * cost_mask).sum() / cost_den
            pred = torch.argmax(cost_logits, dim=1)
            correct = (pred == cost_targets).to(cost_mask.dtype) * cost_mask
            cost_acc = correct.sum() / cost_den

        # [batch_size, seq_len, state_dim]
        if state_preds.shape[1] <= 1:
            state_loss = self._zero_scalar_like(states)
        else:
            state_mask = mask[:, :-1]
            state_mass = float(state_mask.sum().item())
            if state_mass <= 0.0:
                state_loss = self._zero_scalar_like(states)
            else:
                state_mse = F.mse_loss(
                    state_preds[:, :-1], states[:, 1:].detach(), reduction="none"
                ).mean(dim=-1)
                state_den = torch.clamp(state_mask.sum(), min=1.0)
                state_loss = (state_mse * state_mask).sum() / state_den

        loss = act_loss + self.cost_weight * cost_loss + self.state_weight * state_loss

        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optim.step()
        self.scheduler.step()

        metrics = TrainingStepMetrics(
            all_loss=float(loss.item()),
            act_loss=float(act_loss.item()),
            cost_loss=float(cost_loss.item()),
            cost_acc=float(cost_acc.item()),
            state_loss=float(state_loss.item()),
            train_lr=float(self.scheduler.get_last_lr()[0]),
        )
        self.logger.store(
            tab="train",
            all_loss=metrics.all_loss,
            act_loss=metrics.act_loss,
            cost_loss=metrics.cost_loss,
            cost_acc=metrics.cost_acc,
            state_loss=metrics.state_loss,
            train_lr=metrics.train_lr,
        )
        return metrics

    def evaluate(self, num_rollouts, target_return, target_cost):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        if self.env is None:
            raise ValueError("CDTTrainer.evaluate requires a non-None env.")
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(num_rollouts, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(
                self.model, self.env, target_return, target_cost
            )
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return (
            np.mean(episode_rets) / self.reward_scale,
            np.mean(episode_costs) / self.cost_scale,
            np.mean(episode_lens),
        )

    @torch.no_grad()
    def rollout(
        self,
        model: CDT,
        env: Any,
        target_return: float,
        target_cost: float,
    ) -> Tuple[float, float, float]:
        """
        Evaluates the performance of the model on a single episode.
        """
        states = torch.zeros(
            1,
            model.episode_len + 1,
            model.state_dim,
            dtype=torch.float,
            device=self.device,
        )
        actions = torch.zeros(
            1, model.episode_len, dtype=torch.long, device=self.device
        )
        returns = torch.zeros(
            1, model.episode_len + 1, dtype=torch.float, device=self.device
        )
        costs = torch.zeros(
            1, model.episode_len + 1, dtype=torch.float, device=self.device
        )
        time_steps = torch.arange(
            model.episode_len, dtype=torch.long, device=self.device
        )
        time_steps = time_steps.view(1, -1)

        obs, info = env.reset()
        states[:, 0] = torch.as_tensor(obs, device=self.device)
        returns[:, 0] = torch.as_tensor(target_return, device=self.device)
        costs[:, 0] = torch.as_tensor(target_cost, device=self.device)

        epi_cost = torch.tensor(
            np.array([target_cost]), dtype=torch.float, device=self.device
        )

        # cannot step higher than model episode len, as timestep embeddings will crash
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for step in range(model.episode_len):
            # first select history up to step, then select last seq_len states,
            # step + 1 as : operator is not inclusive, last action is dummy with zeros
            # (as model will predict last, actual last values are not important)
            s = states[:, : step + 1][:, -model.seq_len :]
            a = actions[:, : step + 1][:, -model.seq_len :]
            r = returns[:, : step + 1][:, -model.seq_len :]
            c = costs[:, : step + 1][:, -model.seq_len :]
            t = time_steps[:, : step + 1][:, -model.seq_len :]

            action_logits, _, _ = model(s, a, r, c, t, None, epi_cost)
            act = int(torch.argmax(action_logits[0, -1]).item())

            obs_next, reward, terminated, truncated, info = env.step(act)
            if self.cost_reverse:
                cost = (1.0 - info["cost"]) * self.cost_scale
            else:
                cost = info["cost"] * self.cost_scale
            # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
            actions[:, step] = act
            states[:, step + 1] = torch.as_tensor(obs_next)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)
            costs[:, step + 1] = torch.as_tensor(costs[:, step] - cost)

            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"]

            if terminated or truncated:
                break

        return episode_ret, episode_len, episode_cost

    def get_ensemble_action(self, size: int, model, s, a, r, c, t, epi_cost):
        # [size, seq_len, state_dim]
        s = torch.repeat_interleave(s, size, 0)
        # [size, seq_len]
        a = torch.repeat_interleave(a, size, 0)
        # [size, seq_len]
        r = torch.repeat_interleave(r, size, 0)
        c = torch.repeat_interleave(c, size, 0)
        t = torch.repeat_interleave(t, size, 0)
        epi_cost = torch.repeat_interleave(epi_cost, size, 0)

        action_logits, _, _ = model(s, a, r, c, t, None, epi_cost)
        logits = torch.mean(action_logits, dim=0, keepdim=True)
        return int(torch.argmax(logits[0, -1]).item())

    def collect_random_rollouts(self, num_rollouts):
        if self.env is None:
            raise ValueError(
                "CDTTrainer.collect_random_rollouts requires a non-None env."
            )
        episode_rets = []
        for _ in range(num_rollouts):
            obs, info = self.env.reset()
            episode_ret = 0.0
            for step in range(self.model.episode_len):
                act = self.env.action_space.sample()
                obs_next, reward, terminated, truncated, info = self.env.step(act)
                episode_ret += reward
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
        return np.mean(episode_rets)
