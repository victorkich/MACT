from dataclasses import dataclass
from collections import deque

from einops import rearrange, repeat
import dataclasses
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .kv_caching import KeysValues
from .slicer import Embedder, Head, Slicer, DiscreteDist
from .tokenizer import Tokenizer

from .transformer import Transformer, TransformerConfig, get_sinusoid_encoding_table
from .transformer import Perceiver, PerceiverConfig
from .reward_utils import losses_dict

from typing import Tuple, List, Any, Optional
from .world_model_env import MAWorldModelEnv
from utils import init_weights, action_split_into_bins, symlog, obs_split_into_bins


@dataclass
class MAWorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    pred_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    pred_avail_action: torch.FloatTensor
    attn_output: List


def augment_vector(x: torch.Tensor, noise_std: float=0.00, drop_p: float=0.05, clamp: bool=False) -> torch.Tensor:
    drop_mask = (torch.rand_like(x) > drop_p).float()
    noise = torch.randn_like(x) * noise_std
    return torch.clamp(x * drop_mask + noise, x.min(), x.max())


class ContrastiveBlock(nn.Module):
    """Maps (q_in, k_pos) → (q_feat, k_feat) into the same projection space."""
    def __init__(self, in_dim: int, state_dim: int, proj_dim: int = 512):
        super().__init__()
        self.q_mlp = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.ELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.k_mlp = nn.Sequential(
            nn.Linear(state_dim, proj_dim), nn.ELU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, q_in: torch.Tensor, k_pos: torch.Tensor):
        # returns projected query features and projected key features
        return self.q_mlp(q_in), self.k_mlp(k_pos)


class MAWorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, num_action_tokens: int, num_agents: int,
                 config: TransformerConfig, perattn_config: PerceiverConfig,
                 action_dim: int, use_bin: bool = False, bins: int = 64,
                 ### options for continuous action discretization
                 action_bins: int = 256, action_low: float = None, action_high: float = None,
                 combine_action: bool = False,
                 ### options for setting prediction head
                 use_symlog: bool = False, use_ce_for_end: bool = False, use_ce_for_av_action: bool = True,
                 enable_av_pred: bool = False,
                 cpc_mode: str = 'per_agent',  # 'team' or 'per_agent'
                 action_agg: str = 'mean',  # 'mean'|'sum'|'max'
                 detach_keys: bool = False,  # stop grad through positive keys
                 K_cpc: int = 8,  # horizon for CPC
                 cpc_v2: bool = False,         # latent-space aug + stop-grad + temperature
                 cpc_temp: float = 1.0,        # InfoNCE temperature
                 latent_aug_scale: float = 0.1,  # gaussian noise scale (× embedding std)
                 use_ce_for_reward: bool = False, rewards_prediction_config: dict = None) -> None:
        super().__init__()
        # ── AC-CPC v2 options ────────────────────────────────────────────────
        # deviation — MACT's VQ tokenizer quantizes away raw-obs augmentation, unlike
        # TWISTER's continuous CNN encoder, so we augment in embedding space instead.
        self.cpc_v2 = cpc_v2
        self.cpc_temp = cpc_temp        # 1.0 = no temperature (matches TWISTER)
        self.latent_aug_scale = latent_aug_scale
        # detach_keys stays as passed (default False) — TWISTER does not stop-grad keys
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.use_bin = use_bin
        self.bins = bins

        # used for the case when world model needs to deal with continuous actions
        self.use_continuous_action = False or (action_low is not None and action_high is not None)
        self.action_low = action_low
        self.action_high = action_high
        self.action_bins = action_bins
        self.combine_action = combine_action

        self.num_modalities = 3

        self.config = config
        self.num_agents = num_agents

        ## perceiver attention
        self.perattn_config = perattn_config
        self.perattn = Perceiver(**dataclasses.asdict(perattn_config))

        self.agent_id_pos_emb = get_sinusoid_encoding_table(30, perattn_config.dim)

        self.num_action_tokens = num_action_tokens  # for continuous task, this should be dimension of joint action (e.g. like ManiSkill2)
        self.num_obs_tokens = config.tokens_per_block - num_action_tokens - 1  # 其中有一个是perceiver attn的输出

        self.transformer = Transformer(config)

        act_tokens_pattern = torch.zeros(config.tokens_per_block)
        act_tokens_pattern[-1 - num_action_tokens: -1] = 1
        self.act_tokens_pattern = act_tokens_pattern

        obs_tokens_pattern = torch.zeros(config.tokens_per_block)
        obs_tokens_pattern[:self.num_obs_tokens] = 1
        self.obs_tokens_pattern = obs_tokens_pattern

        ### for autoregressive manner
        obs_autoregress_pattern = obs_tokens_pattern.clone()
        obs_autoregress_pattern = torch.roll(obs_autoregress_pattern, -1)

        ### due to attention mask, the last token of transformer output is generated by all tokens of input
        all_but_last_pattern = torch.zeros(config.tokens_per_block)
        all_but_last_pattern[-1] = 1

        ### Perceiver Attention output pattern
        perattn_pattern = torch.zeros(config.tokens_per_block)
        perattn_pattern[-1] = 1
        self.perattn_pattern = perattn_pattern
        self.perattn_slicer = Slicer(max_blocks=config.max_blocks, block_mask=perattn_pattern)
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList(
                [nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        # ---------- AC-CPC additions ----------
        self.D_x = self.config.embed_dim  # Transformer token dim
        self.state_dim = perattn_config.latent_dim # D_e
        self.act_token_dim = 1 # one token / agent / step
        self.cpc_mode = cpc_mode
        self.action_agg = action_agg
        self.detach_keys = detach_keys
        self.K = int(K_cpc)
        self.vector_state = True  # we’re on vector obs; image path would call external self.aug_view

        # Size used for AC context features (per step)
        self.act_feat_dim = (self.act_vocab_size if not self.use_continuous_action
                             else self.action_dim)

        # Contrastive blocks: input dim = Dx + De + k * |A|   (team-agg keeps |A| here)
        self.contrastive_network = nn.ModuleList([
            ContrastiveBlock(
                in_dim=int(self.config.embed_dim + self.perattn_config.latent_dim + k * self.act_feat_dim),
                state_dim=int(self.perattn_config.latent_dim),
                proj_dim=512
            )
            for k in range(self.K)
        ])

        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=obs_autoregress_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )

        ## dense reward predictor
        self.use_symlog = use_symlog  # whether to use symlog transformation
        self.use_ce_for_reward = use_ce_for_reward
        if use_ce_for_reward:
            print("Use cross-entropy to train the prediction of reward...")
        else:
            print("Use SmoothL1Loss to train the prediction of reward...")

        if not self.use_ce_for_reward:
            self.head_rewards = Head(
                max_blocks=config.max_blocks,
                block_mask=all_but_last_pattern,
                head_module=nn.Sequential(
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, 1),
                )
            )

        else:
            assert rewards_prediction_config is not None
            self.use_symlog = True
            bin_width = (rewards_prediction_config["max_v"] - rewards_prediction_config["min_v"]) / \
                        rewards_prediction_config["bins"]
            self.reward_loss = losses_dict[rewards_prediction_config["loss_type"]](
                min_value=rewards_prediction_config["min_v"],
                max_value=rewards_prediction_config["max_v"],
                num_bins=rewards_prediction_config["bins"],
                sigma=bin_width * 0.75
            )
            print(f'Use {self.reward_loss} for discrete labels...')

            self.head_rewards = Head(
                max_blocks=config.max_blocks,
                block_mask=all_but_last_pattern,
                head_module=nn.Sequential(
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, self.reward_loss.output_dim),
                )
            )

        self.use_ce_for_end = use_ce_for_end
        if use_ce_for_end:
            print("Use cross-entropy to train the prediction of termination...")
        else:
            print("Use log-prob to train the prediction of termination...")

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2 if use_ce_for_end else 1),
            )
        )

        self.action_dim = action_dim
        self.enable_av_pred = enable_av_pred
        self.use_ce_for_av_action = use_ce_for_av_action
        ## predict the avail action at next timestep (not current timestep)
        if self.enable_av_pred:
            if use_ce_for_av_action:
                print("Use cross-entropy to train the prediction of av_action...")
            else:
                print("Use log-prob to train the prediction of av_action...")

        else:
            print("Disable the prediction of av_action...")

        if self.enable_av_pred:
            if not self.use_ce_for_av_action:
                self.heads_avail_actions = Head(
                    max_blocks=config.max_blocks,
                    block_mask=all_but_last_pattern,
                    head_module=nn.Sequential(
                        nn.Linear(config.embed_dim, config.embed_dim),
                        nn.ReLU(),
                        nn.Linear(config.embed_dim, config.embed_dim),
                        nn.ReLU(),
                        nn.Linear(config.embed_dim, action_dim),
                    )
                )

            else:
                self.heads_avail_actions = Head(
                    max_blocks=config.max_blocks,
                    block_mask=all_but_last_pattern,
                    head_module=DiscreteDist(
                        config.embed_dim, self.act_vocab_size, 2, 256
                    )
                )

        self.apply(init_weights)

        self.use_ib = False  # use iris databuffer
        if self.use_symlog:
            print("Enable `symlog` to transform the reward targets...")
        else:
            print("Disable `symlog` to transform...")

    def __repr__(self) -> str:
        return "multi_agent_world_model"

    def _reduce_agents(self, x: torch.Tensor, b: int, n: int, how: str = 'mean') -> torch.Tensor:
        """
        x: (B*N, L, D) → returns (B, L, D) by reducing over N.
        """
        x = rearrange(x, '(b n) l d -> b l n d', b=b, n=n)
        if how == 'sum':
            x = x.sum(dim=2)
        elif how == 'max':
            x = x.max(dim=2).values
        else:
            x = x.mean(dim=2)
        return x

    def _team_action_onehot(self, act_tokens: torch.Tensor, A: int, how: str = 'mean') -> torch.Tensor:
        """
        act_tokens: (B, L, N, 1) discrete indices → (B, L, A) aggregated across N.
        """
        act_1h = F.one_hot(act_tokens.squeeze(-1), num_classes=A).float()  # (B, L, N, A)
        if how == 'sum':
            return act_1h.sum(dim=2)  # counts per action
        elif how == 'max':
            return act_1h.max(dim=2).values  # presence per action
        else:
            return act_1h.mean(dim=2)  # frequency per action

    def compute_contrastive_loss(self, feats, embed):
        # Flatten (B*L, D)
        features_x = feats.flatten(start_dim=0, end_dim=1)
        features_y = embed.flatten(start_dim=0, end_dim=1)

        # Similarities (scaled by InfoNCE temperature; cpc_temp=1.0 → original behavior)
        features = features_x.matmul(features_y.transpose(0, 1)) / self.cpc_temp

        # Positives on diagonal
        features_pos = torch.diag(features)
        # LogSumExp over all (incl. positive)
        features_all = torch.logsumexp(features, dim=-1)

        # InfoNCE (pos - logsumexp), so we MINIMIZE the negative later
        info_nce = features_pos - features_all

        # Top-1 accuracy
        with torch.no_grad():
            pred = features.argmax(dim=-1).detach().cpu()
            acc_con = torch.mean((pred == torch.arange(0, features.shape[0])).float())

        return info_nce, acc_con

    def ac_cpc_loss(self, z_clean, h_clean, action_onehot, z_noisy):
        """
        InfoNCE across k = 0..K-1
          k = 0  → same-step (no actions)
          k > 0  → predict k steps ahead, conditioned on concat of next k action one-hots
        Shapes expected: z_clean, h_clean, z_noisy: (B', T, D); action_onehot: (B', T, |A|)
        """
        Bp, T, _ = z_clean.shape
        K_max = min(self.K, T)

        lam = 0.75
        Z = sum(lam ** k for k in range(K_max))
        losses, accs = [], []

        for k in range(K_max):
            if k == 0:
                q_in = torch.cat([h_clean, z_clean], dim=-1)  # (B',T,Dx+De)
                k_pos = z_noisy                              # (B',T,De)
            else:
                L = T - k
                # stack the next k action one-hots along feature dim
                a_ctx = torch.cat([action_onehot[:, i:i+L] for i in range(k)], dim=-1)  # (B',L,k*|A|)
                q_in = torch.cat([h_clean[:, :-k], z_clean[:, :-k], a_ctx], dim=-1)     # (B',L,feat)
                k_pos = z_noisy[:, k:]                                                  # (B',L,De)

            q_feat, k_feat = self.contrastive_network[k](q_in.float(), k_pos.float())

            if q_feat.dtype != torch.float32 or k_feat.dtype != torch.float32:
                with torch.cuda.amp.autocast(enabled=False):
                    vec_loss, acc_k = self.compute_contrastive_loss(q_feat.float(), k_feat.float())
            else:
                vec_loss, acc_k = self.compute_contrastive_loss(q_feat, k_feat)

            # We *minimize* negative of InfoNCE mean
            losses.append(-(vec_loss.mean()) * (lam ** k) / Z)
            accs.append(acc_k)

        return torch.stack(losses).mean(), torch.stack(accs).mean()

    def augment_view(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vector‐state augmentation (dropout + small noise + clamp to observed range).
        """
        # reuse your augment_vector with sensible defaults
        return augment_vector(x, noise_std=0.0, drop_p=0.03, clamp=True)

    def augment_latent(self, x: torch.Tensor) -> torch.Tensor:
        """v2 augmentation: additive Gaussian noise in embedding space, scaled to the
        embedding's own std. Applied AFTER tokenization, so it is not quantized away."""
        std = x.detach().std()
        return x + torch.randn_like(x) * std * self.latent_aug_scale

    def forward(self, tokens: torch.LongTensor, perattn_out: torch.Tensor = None,
                past_keys_values: Optional[KeysValues] = None, return_attn: bool = False,
                attention_mask: torch.Tensor = None) -> MAWorldModelOutput:
        bs = tokens.size(0)
        num_steps = tokens.size(1)  # (B, T)

        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        sequences = self.embedder(tokens, num_steps, prev_steps)

        indices = self.perattn_slicer.compute_slice(num_steps, prev_steps)
        if perattn_out is not None:
            assert len(indices) != 0
            sequences[:, indices] = perattn_out
        else:
            assert len(indices) == 0

        sequences += self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

        x, attn_output = self.transformer(sequences,
                                          past_keys_values,
                                          return_attn=return_attn,
                                          attention_mask=attention_mask)

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        pred_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_avail_action = self.heads_avail_actions(x, num_steps=num_steps,
                                                       prev_steps=prev_steps) if self.enable_av_pred else None

        return MAWorldModelOutput(x, logits_observations, pred_rewards, logits_ends, logits_avail_action,
                                  attn_output=attn_output)

    def compute_loss(self, batch, tokenizer: Tokenizer, attention_mask: torch.Tensor = None, **kwargs: Any):
        device = batch['observation'].device

        # only take discrete action space into account
        if self.use_continuous_action:
            valid_actions = torch.clip(batch['action'], self.action_low, self.action_high)
            act_tokens = action_split_into_bins(valid_actions, self.action_bins, self.action_low,
                                                self.action_high)  # (B L K A)
            if self.combine_action:
                combined_act_tokens = rearrange(act_tokens, 'b t n a -> b t (n a)')
                combined_act_tokens = repeat(combined_act_tokens, 'b t a -> b t n a', n=self.num_agents)
        else:
            act_tokens = torch.argmax(batch['action'], dim=-1, keepdim=True)

        ### modified for ablation ###
        if not self.use_bin:
            with torch.no_grad():
                ### when tokenizer is `Tokenizer` run these two lines
                # tokenizer_encodings = tokenizer.encode(batch['observation'], should_preprocess=True)  # (B, L, K)
                # obs_tokens = tokenizer_encodings.tokens

                ### when tokenizer is `SimpleVQAutoEncoder` run these two lines
                obs_t_embeds, obs_tokens = tokenizer.encode(batch['observation'], should_preprocess=True)
                obs_tokens = obs_tokens.to(torch.long)
        else:
            observations = symlog(batch['observation'])
            obs_tokens = obs_split_into_bins(observations, self.bins, low=-3., high=3.)
        ### --------------------- ###

        obs_encodings = self.embedder.embedding_tables[1](obs_tokens)
        action_encodings = self.embedder.embedding_tables[0](act_tokens)
        input_encodings = torch.cat([obs_encodings, action_encodings], dim=-2)

        b, l, N, M, e = input_encodings.shape

        agent_id_emb = repeat(self.agent_id_pos_emb[:, :self.num_agents], '1 n e -> (b l) (n m) e', b=b, l=l, m=M)
        input_encodings = rearrange(input_encodings, 'b l n m e -> (b l) (n m) e') + agent_id_emb.detach().to(device)

        perattn_out = self.perattn(input_encodings)
        perattn_out = rearrange(perattn_out, '(b l) n e -> (b n) l e', b=b, l=l, n=N)

        if self.combine_action:
            tokens = torch.cat([obs_tokens, combined_act_tokens, torch.empty(*combined_act_tokens.shape[:-1], 1, device=device, dtype=torch.long)], dim=-1)  # (B, L, (K+N))

        else:
            tokens = torch.cat([obs_tokens, act_tokens, torch.empty(*act_tokens.shape[:-1], 1, device=device, dtype=torch.long)], dim=-1)  # (B, L, (K+N))

        tokens = rearrange(tokens.transpose(1, 2), 'b n l k -> (b n) (l k)')  # (B, L(K+N))

        outputs = self(tokens, perattn_out=perattn_out, attention_mask=attention_mask)

        # compute labels
        if self.use_ib:  # if use iris databuffer (Deprecated)
            valid_mask = batch['filled'].clone().unsqueeze(-1).expand(-1, -1, self.num_agents).to(torch.float32)

            labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['reward'], batch['done'], batch['filled'])
            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b l o -> (b l) o')

            loss_obs = F.cross_entropy(logits_observations, labels_observations)

            if not self.use_classification:
                pred_ends = td.independent.Independent(td.Bernoulli(logits=outputs.logits_ends), 1)
                loss_ends = -(pred_ends.log_prob((1. - labels_ends)) * valid_mask).sum() / valid_mask.sum()
            else:
                raise NotImplementedError

            l1_criterion = nn.SmoothL1Loss(reduction="none")

            ## regression label for rewards
            labels_rewards = symlog(batch['reward'])

            loss_rewards = l1_criterion(outputs.pred_rewards, labels_rewards)
            loss_rewards = (loss_rewards.squeeze(-1) * valid_mask).sum() / valid_mask.sum()

            pred_av_actions = td.independent.Independent(td.Bernoulli(logits=outputs.pred_avail_action[:, :-1]), 1)
            loss_av_actions = -(
                        pred_av_actions.log_prob(batch['av_action'][:, 1:]) * valid_mask[:, 1:]).sum() / valid_mask[:,
                                                                                                         1:].sum()

        else:  # use mamba databuffer
            ### guided by dones mask, compute observation loss
            dones = rearrange(batch['done'], 'b l n 1 -> (b n) l')
            labels_obs_token = rearrange(obs_tokens, 'b l n m -> (b n) (l m)')
            loss_obs = 0.
            for idx in range(dones.size(0)):
                cur_done = dones[idx]
                if cur_done[:-1].sum() > 0:
                    done_indices = (cur_done[:-1] == 1).nonzero().squeeze(-1) + 1
                    done_indices = done_indices.to(torch.long)

                    cur_loss = 0.
                    last_idx = 0
                    for k in done_indices.tolist():
                        last_divide_idx = last_idx * self.num_obs_tokens
                        divide_idx = k * self.num_obs_tokens
                        cur_loss += F.cross_entropy(outputs.logits_observations[idx, last_divide_idx: (divide_idx - 1)],
                                                    labels_obs_token[idx, (last_divide_idx + 1): divide_idx])

                        last_idx = k

                    last_divide_idx = last_idx * self.num_obs_tokens
                    cur_loss += F.cross_entropy(outputs.logits_observations[idx, last_divide_idx: -1],
                                                labels_obs_token[idx, (last_divide_idx + 1):])
                    cur_loss /= (len(done_indices.tolist()) + 1)
                    loss_obs += cur_loss

                else:
                    loss_obs += F.cross_entropy(outputs.logits_observations[idx, :-1], labels_obs_token[idx, 1:])

            loss_obs /= dones.size(0)

            ### compute discount loss
            if not self.use_ce_for_end:
                pred_ends = td.independent.Independent(td.Bernoulli(logits=outputs.logits_ends), 1)
                loss_ends = -torch.mean(pred_ends.log_prob((1. - rearrange(batch['done'], 'b l n 1 -> (b n) l 1'))))
            else:
                logits_ends = rearrange(outputs.logits_ends, 'b l e -> (b l) e')
                labels_ends = rearrange(batch['done'], 'b l n 1 -> (b n l)').to(torch.long)
                loss_ends = F.cross_entropy(logits_ends, labels_ends)

            ### compute reward loss
            labels_rewards = rearrange(batch['reward'], 'b l n 1 -> (b n) l 1')
            if self.use_symlog:
                labels_rewards = symlog(labels_rewards)

            if self.use_ce_for_reward:
                labels_rewards = rearrange(labels_rewards, 'b l 1 -> (b l 1)')
                logits_rewards = rearrange(outputs.pred_rewards, 'b l e -> (b l) e')
                loss_rewards = self.reward_loss(logits_rewards, labels_rewards)

            else:
                loss_rewards = F.smooth_l1_loss(outputs.pred_rewards, labels_rewards)

            ### compute av_action loss
            if self.enable_av_pred:
                tmp = torch.roll(batch['done'], 1, dims=1).squeeze(-1)
                labels_av_actions = batch['av_action']
                labels_av_actions[tmp == True] = torch.ones_like(labels_av_actions[tmp == True], device=device)

                ## for cross-entropy loss
                if self.use_ce_for_av_action:
                    logits_av_actions = rearrange(outputs.pred_avail_action[:, :-1], 'b l a e -> (b l a) e')
                    labels_av_actions = rearrange(labels_av_actions, 'b l n e -> (b n) l e')[:, 1:].reshape(-1, ).to(
                        torch.long)
                    loss_av_actions = F.cross_entropy(logits_av_actions, labels_av_actions)

                else:
                    pred_av_actions = td.independent.Independent(td.Bernoulli(logits=outputs.pred_avail_action[:, :-1]),
                                                                 1)
                    labels_av_actions = rearrange(labels_av_actions, 'b l n e -> (b n) l e')
                    loss_av_actions = -torch.mean(pred_av_actions.log_prob(labels_av_actions[:, 1:]))
            else:
                loss_av_actions = 0.

        z_agents = perattn_out  # (B*N, L, De)

        # indices of Perceiver token positions inside Transformer blocks → pick h at those slots
        num_steps_tok = tokens.size(1)
        per_indices = self.perattn_slicer.compute_slice(num_steps_tok, 0)
        h_agents = outputs.output_sequence[:, per_indices]  # (B*N, L, Dx)

        # Actions → one-hot
        A = (self.act_vocab_size if not self.use_continuous_action else self.action_dim)
        act_1h_agents = F.one_hot(act_tokens.squeeze(-1), num_classes=A).float()  # (B, L, N, A)

        # Augmented positive keys: re-run Perceiver on noised view
        if self.cpc_v2:
            # v2: perturb the CLEAN obs embeddings in latent space (no re-tokenization),
            # so the augmentation actually survives to the contrastive head.
            obs_enc_aug = self.augment_latent(obs_encodings)  # (B, L, N, M, E)
            aug_in = torch.cat([obs_enc_aug, action_encodings], dim=-2)  # (B,L,N,M+1,E)
        else:
            obs_aug = self.augment_view(batch['observation'])  # (B, L, N, F)
            obs_t_embeds_aug, obs_tokens_aug = tokenizer.encode(obs_aug, should_preprocess=True)
            obs_enc_aug = self.embedder.embedding_tables[1](obs_tokens_aug)  # (B, L, N, M, E)
            aug_in = torch.cat([obs_enc_aug, self.embedder.embedding_tables[0](act_tokens)], dim=-2)  # (B,L,N,M+1,E)

        bb, ll, NN, MM, ee = aug_in.shape
        agent_id_emb_aug = repeat(self.agent_id_pos_emb[:, :self.num_agents], '1 n e -> (b l) (n m) e', b=bb, l=ll,
                                  m=MM).to(aug_in.device)
        aug_in = rearrange(aug_in, 'b l n m e -> (b l) (n m) e') + agent_id_emb_aug.detach()
        z_aug_agents = self.perattn(aug_in)  # ((B*L), N, De)
        z_aug_agents = rearrange(z_aug_agents, '(b l) n e -> (b n) l e', b=bb, l=ll, n=NN)  # (B*N, L, De)

        # Team aggregated and per-agent conditioning
        if self.cpc_mode == 'team':
            # reduce across agents to get one representation per team & timestep
            z_clean = self._reduce_agents(z_agents, b=b, n=N, how=self.action_agg)  # (B, L, De)
            h_clean = self._reduce_agents(h_agents, b=b, n=N, how='mean')  # (B, L, Dx)  (keep mean here)
            z_noisy = self._reduce_agents(z_aug_agents, b=b, n=N, how=self.action_agg)  # (B, L, De)
            act_1hot = self._team_action_onehot(act_tokens, A=A, how=self.action_agg)  # (B, L, A)
        else:
            # original per-agent CPC
            z_clean = z_agents  # (B*N, L, De)
            h_clean = h_agents  # (B*N, L, Dx)
            z_noisy = z_aug_agents  # (B*N, L, De)
            act_1hot = rearrange(act_1h_agents, 'b l n a -> (b n) l a')  # (B*N, L, A)

        if self.detach_keys:
            z_noisy = z_noisy.detach()

        # Compute CPC loss on aggregated tensors
        loss_cpc, acc_cpc = self.ac_cpc_loss(
            z_clean=z_clean,  # (B, L, De) or (B*N, L, De)
            h_clean=h_clean,  # (B, L, Dx) or (B*N, L, Dx)
            action_onehot=act_1hot,  # (B, L, A)  or (B*N, L, A)
            z_noisy=z_noisy  # (B, L, De) or (B*N, L, De)
        )

        # total wm loss
        loss = loss_obs + loss_ends + loss_rewards + loss_av_actions + loss_cpc

        loss_dict = {
            'world_model/loss_obs': loss_obs.item(),
            'world_model/loss_rewards': loss_rewards.item(),
            'world_model/loss_ends': loss_ends.item(),
            'world_model/loss_av_actions': loss_av_actions.item() if self.enable_av_pred else 0.,
            'world_model/loss_cpc': loss_cpc.item(),
            'world_model/acc_cpc': acc_cpc.item(),
            'world_model/total_loss': loss.item(),
        }

        return loss, loss_dict

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor,
                                   filled: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(filled)
        labels_observations = rearrange(
            obs_tokens.masked_fill(mask_fill.unsqueeze(-1).unsqueeze(-1).expand_as(obs_tokens), -100).transpose(1, 2),
            'b n l k -> (b n) (l k)')[:, 1:]

        labels_rewards = rewards.masked_fill(mask_fill.unsqueeze(-1).unsqueeze(-1).expand_as(rewards), 0.)

        labels_ends = ends.masked_fill(mask_fill.unsqueeze(-1).unsqueeze(-1).expand_as(ends), 1.).to(torch.long)

        return labels_observations.reshape(-1), labels_rewards, labels_ends

    def compute_labels_world_model_all_valid(self, obs_tokens: torch.Tensor, rewards: torch.Tensor,
                                             ends: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        labels_observations = rearrange(obs_tokens.transpose(1, 2), 'b n l k -> (b n) (l k)')[:, 1:]
        labels_rewards = rearrange(rewards.transpose(1, 2), 'b n l 1 -> (b n) l 1')
        labels_ends = rearrange(ends.transpose(1, 2), 'b n l 1 -> (b n) l 1')
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1).to(torch.long)

    def get_perceiver_attn_out(self, obs_tokens, actions):
        device = obs_tokens.device
        shape = obs_tokens.shape

        obs_encodings = self.embedder.embedding_tables[1](obs_tokens)
        action_encodings = self.embedder.embedding_tables[0](actions)
        input_encodings = torch.cat([obs_encodings, action_encodings], dim=-2)

        n, m, e = input_encodings.shape[-3:]
        input_encodings = rearrange(input_encodings, '... n m e -> (...) (n m) e')
        agent_id_emb = repeat(self.agent_id_pos_emb[:, :n], '1 n e -> b (n m) e', b=input_encodings.size(0), n=n, m=m)

        input_encodings += agent_id_emb.detach().to(device)
        perattn_out = self.perattn(input_encodings)

        perattn_out = perattn_out.reshape(*shape[:-1], -1)
        return perattn_out

    def get_perceiver_cross_attn_w(self, obs_tokens, actions):
        device = obs_tokens.device
        shape = obs_tokens.shape

        obs_encodings = self.embedder.embedding_tables[1](obs_tokens)
        action_encodings = self.embedder.embedding_tables[0](actions)
        input_encodings = torch.cat([obs_encodings, action_encodings], dim=-2)

        n, m, e = input_encodings.shape[-3:]
        input_encodings = rearrange(input_encodings, '... n m e -> (...) (n m) e')
        agent_id_emb = repeat(self.agent_id_pos_emb[:, :n], '1 n e -> b (n m) e', b=input_encodings.size(0), n=n, m=m)

        input_encodings += agent_id_emb.detach().to(device)
        perattn_out, cross_attn_w = self.perattn(input_encodings, return_cross_attn=True)

        perattn_out = perattn_out.reshape(*shape[:-1], -1)
        return perattn_out, cross_attn_w

    ### visualize attention map
    @torch.no_grad()
    def visualize_attn(self, sample, tokenizer, save_dir):
        # preliminary
        device = sample["observation"].device
        n_agents = sample['observation'].shape[-2]
        horizon = sample['observation'].shape[-3]
        obs_token_indices = rearrange(repeat(self.obs_tokens_pattern, 'n -> h n', h=horizon), 'h n -> (h n)')
        obs_token_indices = (obs_token_indices == 1).nonzero().squeeze().numpy()
        act_token_indices = rearrange(repeat(self.act_tokens_pattern, 'n -> h n', h=horizon), 'h n -> (h n)')
        act_token_indices = (act_token_indices == 1).nonzero().squeeze().numpy()
        perattn_indices = rearrange(repeat(self.perattn_pattern, 'n -> h n', h=horizon), 'h n -> (h n)')
        perattn_indices = (perattn_indices == 1).nonzero().squeeze().numpy()

        save_dir.mkdir(parents=True, exist_ok=True)
        for agent_id in range(n_agents):
            tmp_dir = save_dir / f"agent_{agent_id}"
            tmp_dir.mkdir(parents=True, exist_ok=True)

        for horizon_idx in range(horizon):
            tmp_dir = save_dir / f"horizon_{horizon_idx}"
            tmp_dir.mkdir(parents=True, exist_ok=True)

        _, obs_tokens = tokenizer.encode(sample['observation'], should_preprocess=True)
        obs_tokens = obs_tokens.to(torch.long)
        act_tokens = torch.argmax(sample['action'], dim=-1, keepdim=True)

        perattn_out = self.get_perceiver_attn_out(obs_tokens, act_tokens)
        b, l, n, e = perattn_out.shape
        perattn_out = rearrange(perattn_out, 'b l n e -> (b n) l e', b=b, l=l, n=n)

        tokens = torch.cat([obs_tokens, act_tokens, torch.empty_like(act_tokens, device=device, dtype=torch.long)],
                           dim=-1)
        tokens = rearrange(tokens.transpose(1, 2), 'b n l k -> (b n) (l k)')  # (B, L(K+N))

        outputs = self(tokens, perattn_out=perattn_out, return_attn=True)

        ### visualize perceiver_cross_attn
        _, cross_attn_weight = self.get_perceiver_cross_attn_w(obs_tokens, act_tokens)
        cross_attn_weight = cross_attn_weight.cpu().numpy()

        attn_output = outputs.attn_output

        # define custom cmap
        # modality_colors = ["Blues", "Reds", "Oranges"]
        modality_colors = ["Blues", "Reds", "YlOrBr"]
        # modality_colors = ['#8DC7E3', '#FF988C', '#FFC995']
        colors = []
        for color in modality_colors:
            cmap = mpl.colormaps[color]
            colors.append(
                cmap(np.linspace(0., 1., 333))
            )

        white_cmap = LinearSegmentedColormap.from_list("white", [(0., 'white'), (1., 'white')], N=1)
        colors.append(
            white_cmap(np.linspace(0., 1., 1))
        )

        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", np.vstack(colors))
        red_cmap = mpl.colormaps["Oranges"]

        def save_matrix_as_image(matrix, filename, custom_cmap):
            plt.imshow(matrix, cmap=custom_cmap, vmin=0, vmax=1)
            # plt.colorbar(orientation="horizontal")

            plt.axis("off")

            # indices = np.tril_indices_from(matrix)

            # min_row, max_row = min(indices[0]), max(indices[0])
            # min_col, max_col = min(indices[1]), max(indices[1])

            # vertices = [(min_col - 0.5, min_row - 0.5), (max_col + 0.5, min_row - 0.5), (max_col + 0.5, max_row + 0.5)]

            # triangle = plt.Polygon(vertices, edgecolor='black', linewidth=2, fill=None)
            # plt.gca().add_patch(triangle)
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=600)

            plt.close()

        import seaborn as sns
        import pandas as pd
        import matplotlib.patches as patches

        square_size = 20

        fig_width = cross_attn_weight.shape[-1] * square_size / 100
        fig_height = cross_attn_weight.shape[-2] * square_size / 100

        for horizon_idx in range(cross_attn_weight.shape[0]):
            for head_id in range(cross_attn_weight.shape[1]):
                matrix = cross_attn_weight[horizon_idx, head_id]

                df = pd.DataFrame(matrix, index=[None for i in range(1, n_agents + 1)])

                df.columns = [None] * len(df.columns)

                # plt.imshow(matrix, cmap='viridis')
                fig = plt.figure(figsize=(fig_width, fig_height));
                heatmap = sns.heatmap(df, vmin=-0.05, vmax=1.05, cmap=sns.cubehelix_palette(as_cmap=True), square=True,
                                      cbar_kws={'aspect': 5})

                heatmap.set_xticks(np.arange(0.5, len(matrix[0]), 1))
                heatmap.set_yticks(np.arange(0.5, len(matrix), 1))

                plt.gca().patch.set_edgecolor('black');
                plt.gca().patch.set_linewidth('1')

                cax = plt.gcf().axes[-1];
                cax.set_frame_on(True);
                cax.patch.set_edgecolor('black');
                cax.patch.set_linewidth(
                    '1')  # cax.add_patch(patches.Rectangle((-0.05, -0.05), 1.05, 1.05, fill=False, edgecolor='black', linewidth=2))

                plt.savefig(save_dir / f"horizon_{horizon_idx}" / f"cross_attn_head{head_id}.png",
                            bbox_inches="tight", pad_inches=0.1, dpi=600)
                plt.close()

        ## save as image
        scale = 0.332
        for layer_id in range(len(attn_output)):
            attn_weight = attn_output[layer_id].cpu().numpy()
            attn_weight[:, :, obs_token_indices] *= scale

            attn_weight[:, :, act_token_indices] *= scale
            attn_weight[:, :, act_token_indices] += 0.3335

            attn_weight[:, :, perattn_indices] *= scale
            attn_weight[:, :, perattn_indices] += 0.6665

            attn_weight = np.where(np.tril(np.ones_like(attn_weight)) == 1, attn_weight,
                                   np.zeros_like(attn_weight) + 0.9995)

            for agent_id in range(attn_weight.shape[0]):
                for head_id in range(attn_weight.shape[1]):
                    save_matrix_as_image(attn_weight[agent_id, head_id],
                                         save_dir / f"agent_{agent_id}" / f"layer{layer_id}_head{head_id}.png",
                                         custom_cmap)

        print(f"Attention visualization has been saved to {str(save_dir)}.")


def rollout_policy_trans(wm_env: MAWorldModelEnv, policy, critic, horizons, observations, av_actions, filled, **kwargs):
    use_stack = kwargs.get("use_stack", False)

    init_obs = observations[:, -1].clone()
    av_action = av_actions[:, -1].clone() if av_actions is not None else None

    if use_stack:
        stack_obs_num = kwargs.get("stack_obs_num", None)
        assert stack_obs_num is not None and type(stack_obs_num) == int

        stack_obs = deque(maxlen=stack_obs_num)

        tmp_obs = observations[:, :-1].clone()
        tmp_filled = filled[:, :-1, None, None].clone().repeat(1, 1, *tmp_obs.shape[-2:])
        unvalid_obs = torch.zeros_like(tmp_obs, device=tmp_obs.device)
        tmp_obs = wm_env.tokenizer.encode_decode(tmp_obs, True, True)
        tmp_obs = torch.where(tmp_filled == True, tmp_obs, unvalid_obs)

        for index in range(stack_obs_num - 1):
            stack_obs.append(tmp_obs[:, index])

    actor_feats = []
    critic_feats = []
    actions = []
    av_actions = []
    policies = []
    rewards = []
    dones = []

    values = []

    # initialize wm_env
    rec_obs, critic_feat = wm_env.reset_from_initial_observations(init_obs)

    for t in range(horizons):
        if use_stack:
            stack_obs.append(rec_obs)
            feat = rearrange(torch.stack(list(stack_obs), dim=0), 'm b n e -> b n (m e)')

        else:
            feat = rec_obs

        #### update for supporting MPE, GRF
        if wm_env.use_continuous_action:
            action, pi = policy(feat, deterministic=False)  # pi -> log_probs
        else:
            _, pi = policy(feat)

            if av_action is not None:
                pi[av_action == 0] = -1e10
                av_actions.append(av_action.squeeze(0))

            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample().squeeze(0)

        value = critic(feat)

        actor_feats.append(feat)
        policies.append(pi)
        actions.append(action)
        critic_feats.append(feat)
        values.append(value)

        if wm_env.use_continuous_action:
            rec_obs, reward, done, av_action, critic_feat = wm_env.step(action,
                                                                        should_predict_next_obs=(t < horizons - 1))
        else:
            rec_obs, reward, done, av_action, critic_feat = wm_env.step(torch.argmax(action, dim=-1).unsqueeze(-1),
                                                                        should_predict_next_obs=(t < horizons - 1))

        rewards.append(reward)
        dones.append(done)

    return {"actor_feats": torch.stack(actor_feats, dim=0),  # torch.stack(actor_feats, dim=0),
            "critic_feats": torch.stack(critic_feats, dim=0),
            "actions": torch.stack(actions, dim=0),
            "av_actions": torch.stack(av_actions, dim=0) if len(av_actions) > 0 else None,
            "old_policy": torch.stack(policies, dim=0),
            "old_values": torch.stack(values, dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "discounts": torch.stack(dones, dim=0),
            }
