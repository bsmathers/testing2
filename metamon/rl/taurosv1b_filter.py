"""Numerically robust advantage filter for TaurosV1B.

Kept separate from ``taurosv1b_online`` so gin can import the filter without
re-executing the online runner when it is launched through ``python -m``.
"""

from __future__ import annotations

import gin
import torch

from metamon.rl.custom_agent import ISAdvantageFilter


@gin.configurable
class RobustAdvantageFilter(ISAdvantageFilter):
    """Normalized exponential AWR filter with safe masked statistics.

    No behavior-policy correction is used unless a caller explicitly injects
    ``delta_log``; TaurosV1B uses this as AWR/CRR-style regression.
    """

    def __call__(self, adv: torch.Tensor) -> torch.Tensor:
        mask = self._mask
        self._mask = None
        delta_log = self._delta_log
        self._delta_log = None

        adv_f = adv.float()
        if mask is not None:
            mask = mask[:, : adv.shape[1], ...]
            while mask.ndim < adv.ndim:
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(adv).bool()
            valid = adv_f[mask]
        else:
            valid = adv_f.reshape(-1)

        if valid.numel() == 0:
            mu = torch.zeros((), device=adv.device, dtype=adv_f.dtype)
            sigma = torch.ones((), device=adv.device, dtype=adv_f.dtype)
        else:
            mu = valid.mean()
            if valid.numel() <= 1:
                sigma = torch.ones_like(mu)
            else:
                sigma = valid.std(unbiased=False)
                sigma = torch.where(
                    torch.isfinite(sigma) & (sigma > self.eps),
                    sigma,
                    torch.ones_like(sigma),
                )

        exponent = self.beta * ((adv_f - mu) / sigma)
        if delta_log is not None:
            delta_log = delta_log[:, : adv.shape[1], ...].float()
            exponent = exponent + torch.clamp(
                delta_log, -self.clip_delta, self.clip_delta
            )
        exponent = torch.nan_to_num(exponent, nan=0.0, posinf=20.0, neginf=-20.0)

        weights = torch.exp(exponent).to(dtype=adv.dtype)
        if self.clip_weights_low is not None or self.clip_weights_high is not None:
            weights = torch.clamp(
                weights, min=self.clip_weights_low, max=self.clip_weights_high
            )
        if self.seq_enabled:
            weights = weights * self._compute_seq_weights(adv)
        return weights
