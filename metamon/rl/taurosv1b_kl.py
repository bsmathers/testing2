"""KL-anchored TaurosV1B Phase-E agent.

The online policy is initialized from Phase C and penalized for moving its action
distribution too far from that frozen Phase-C policy during the first part of
training.  The anchor includes the timestep encoder, trajectory encoder, and
actor, so the regularizer constrains the actual policy distribution rather than
only actor-head weights.
"""
from __future__ import annotations

import copy
from typing import Optional

import gin
import torch
from einops import repeat

import amago
from amago.agent import MultiTaskAgent


_ANCHOR_MODULES = {
    "_kl_anchor_tstep_encoder": "tstep_encoder",
    "_kl_anchor_traj_encoder": "traj_encoder",
    "_kl_anchor_actor": "actor",
}


@gin.configurable
class KLAnchoredMultiTaskAgent(MultiTaskAgent):
    """MultiTaskAgent with a frozen initial-policy KL regularizer."""

    def __init__(
        self,
        *args,
        kl_coeff: float = 1.0,
        kl_anneal_steps: int = 100_000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kl_coeff = float(kl_coeff)
        self.kl_anneal_steps = int(kl_anneal_steps)
        self._kl_anchor_tstep_encoder = copy.deepcopy(self.tstep_encoder)
        self._kl_anchor_traj_encoder = copy.deepcopy(self.traj_encoder)
        self._kl_anchor_actor = copy.deepcopy(self.actor)
        for name in _ANCHOR_MODULES:
            getattr(self, name).requires_grad_(False)
        self.register_buffer("_kl_forward_step", torch.zeros((), dtype=torch.long))

    @staticmethod
    def _copy(dst, src) -> None:
        dst.load_state_dict(src.state_dict(), strict=True)
        dst.requires_grad_(False)
        dst.eval()

    def train(self, mode: bool = True):
        # The anchor must remain deterministic even when the online agent enters
        # training mode; otherwise dropout would make the KL target itself move.
        result = super().train(mode)
        for name in _ANCHOR_MODULES:
            getattr(self, name).eval()
        return result

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        has_anchor = any(k.startswith("_kl_anchor_") for k in state_dict)
        if not has_anchor:
            extra = {}
            for anchor_attr, online_attr in _ANCHOR_MODULES.items():
                prefix = online_attr + "."
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        extra[anchor_attr + k[len(online_attr):]] = v.clone()
            extra["_kl_forward_step"] = torch.zeros_like(self._kl_forward_step)
            state_dict = {**state_dict, **extra}
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def on_checkpoint_loaded(self, is_resume: bool = False):
        if not is_resume:
            # Raw Phase-C bootstrap: make the KL reference exactly the loaded C policy.
            for anchor_attr, online_attr in _ANCHOR_MODULES.items():
                self._copy(getattr(self, anchor_attr), getattr(self, online_attr))
            self._kl_forward_step.zero_()
        for name in _ANCHOR_MODULES:
            getattr(self, name).requires_grad_(False)
            getattr(self, name).eval()

    def _anchor_dist(self, batch):
        straight = {k: batch.obs[k] for k in self.pass_obs_keys_to_actor}
        with torch.no_grad():
            o = self._kl_anchor_tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
            s, _ = self._kl_anchor_traj_encoder(
                seq=o, time_idxs=batch.time_idxs, hidden_state=None
            )
            return self._kl_anchor_actor(s, straight_from_obs=straight)

    def forward(self, batch, log_step: bool):
        captured = {}

        def capture_actor(_module, _inputs, output):
            captured["dist"] = output

        handle = self.actor.register_forward_hook(capture_actor)
        try:
            total_loss = super().forward(batch, log_step)
        finally:
            handle.remove()

        step = int(self._kl_forward_step.item())
        self._kl_forward_step.add_(1)
        if self.kl_coeff <= 0.0 or self.kl_anneal_steps <= 0 or step >= self.kl_anneal_steps:
            if log_step:
                self.update_info["KL Anchor Coeff"] = torch.zeros(
                    (), device=batch.rl2s.device
                )
            return total_loss

        current = captured["dist"]
        anchor = self._anchor_dist(batch)
        if not self.discrete:
            raise NotImplementedError("KLAnchoredMultiTaskAgent currently expects discrete actions")

        p0 = anchor.probs.detach().float()
        p = current.probs.float()
        eps = 1e-8
        kl = (
            p0
            * (
                torch.log(p0.clamp_min(eps))
                - torch.log(p.clamp_min(eps))
            )
        ).sum(dim=-1, keepdim=True)
        kl = kl[:, :-1, ...]

        state_mask = (~((batch.rl2s == self.pad_val).all(-1, keepdim=True))).bool()[
            :, 1:, ...
        ]
        mask = repeat(state_mask, "b l 1 -> b l g 1", g=len(self.gammas))
        kl_loss = amago.utils.masked_avg(kl, mask)

        frac = max(1.0 - step / float(self.kl_anneal_steps), 0.0)
        coeff = self.kl_coeff * frac
        total_loss = total_loss + coeff * kl_loss

        if log_step:
            self.update_info["KL Anchor Loss"] = kl_loss.detach()
            self.update_info["KL Anchor Coeff"] = torch.tensor(
                coeff, device=kl_loss.device
            )
            self.update_info["KL Anchor Forward Step"] = self._kl_forward_step.detach()
        return total_loss
