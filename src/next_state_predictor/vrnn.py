"""Variational Recurrent Neural Network (VRNN) for next-state prediction."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn as nn


class VRNN(nn.Module):
    """Variational Recurrent Neural Network next-state predictor.

    Based on Chung et al. (2015): "A Recurrent Latent Variable Model for
    Sequential Data".  Trained to minimise the negative ELBO — a tractable
    upper bound on the negative log-likelihood.

    At each time step the model maintains a GRU hidden state ``h`` that
    conditions both the prior ``p(z|h)`` and the approximate posterior
    ``q(z|x,h)``.  The hidden state is then updated from ``(phi_x(x_t),
    phi_z(z_t))``.  After processing the full input sequence a Gaussian
    decoder produces the predicted next state from the final ``(z, h)``.

    During *training* the posterior is computed from the next-step
    observations ``x_1`` (teacher forcing), so the model learns a richer
    approximate posterior.  During *inference* (``forward``) latent samples
    are drawn from the prior, which requires only the current observations.

    Args:
        input_dim: Feature dimension of each observed time step.
        hidden_size: GRU hidden state size and internal feature width.
        latent_dim: Dimension of the latent space.
        output_dim: Dimension of the predicted next state.
        device: PyTorch device.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 64,
        latent_dim: int = 16,
        output_dim: int = 1,
        device=None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.device = device

        # Feature extractors
        self.phi_x = nn.Sequential(
            nn.Linear(input_dim, hidden_size, device=device),
            nn.ReLU(),
        )
        self.phi_z = nn.Sequential(
            nn.Linear(latent_dim, hidden_size, device=device),
            nn.ReLU(),
        )

        # Prior: p(z_t | h_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, device=device),
            nn.Tanh(),
        )
        self.prior_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.prior_log_var = nn.Linear(hidden_size, latent_dim, device=device)

        # Encoder: q(z_t | x_t, h_{t-1})
        self.enc_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, device=device),
            nn.Tanh(),
        )
        self.enc_mean = nn.Linear(hidden_size, latent_dim, device=device)
        self.enc_log_var = nn.Linear(hidden_size, latent_dim, device=device)

        # Decoder for next-state prediction: p(y | z, h)
        self.dec_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, device=device),
            nn.Tanh(),
        )
        self.dec_mean = nn.Linear(hidden_size, output_dim, device=device)
        self.dec_log_var = nn.Linear(hidden_size, output_dim, device=device)

        # GRU updates h after each step using (phi_x(x_t), phi_z(z_t))
        self.rnn = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            device=device,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_parameters(self) -> Iterator[nn.Parameter]:
        """Return an iterator over all trainable parameters."""
        return self.parameters()

    def get_loss(self, x: torch.Tensor, x_1: torch.Tensor, y: torch.Tensor) -> float:
        """Compute the negative ELBO loss without updating parameters.

        Args:
            x: Current-step observations ``(batch, seq_len, input_dim)``.
            x_1: Next-step observations ``(batch, seq_len, input_dim)``.
            y: Target next states ``(batch, 1, output_dim)`` or
                ``(batch, output_dim)``.

        Returns:
            Scalar loss value.
        """
        return self.train_step(x, x_1, y, optimizer=None)

    def train_step(
        self,
        x: torch.Tensor,
        x_1: torch.Tensor,
        y: torch.Tensor,
        optimizer,
    ) -> float:
        """Perform a single training step (or evaluation if *optimizer* is None).

        The loss is the negative ELBO:

        .. math::
            \\mathcal{L} = \\text{NLL}(y; \\mu_y, \\sigma_y^2)
                         + \\frac{1}{T}\\sum_t \\text{KL}[q(z_t|x_t,h_{t-1})
                                                          \\| p(z_t|h_{t-1})]

        Args:
            x: Current-step observations ``(batch, seq_len, input_dim)``.
            x_1: Next-step observations ``(batch, seq_len, input_dim)``
                used to compute the posterior.
            y: Target next states ``(batch, 1, output_dim)`` or
                ``(batch, output_dim)``.
            optimizer: A PyTorch optimiser, or ``None`` for evaluation only.

        Returns:
            Scalar loss value (negative ELBO).
        """
        if optimizer is not None:
            optimizer.zero_grad()

        h, kl = self._process_sequence(x, x_1)
        _, y_mean, y_log_var = self._decode_next_state(h)

        y_target = y.reshape(x.size(0), -1)[:, : self.output_dim]
        rec_nll = self._gaussian_nll(y_target, y_mean, y_log_var)
        loss = rec_nll + kl

        if optimizer is not None:
            loss.backward()
            optimizer.step()
        return loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a next-state prediction from observations ``x``.

        Latent samples are drawn from the prior (no posterior needed).

        Args:
            x: Observations ``(batch, seq_len, input_dim)``.

        Returns:
            Predicted next state ``(batch, output_dim)``.
        """
        h, _ = self._process_sequence(x, x_1=None)
        _, y_mean, _ = self._decode_next_state(h)
        return y_mean

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reparameterize(
        self, mean: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mean + torch.randn_like(std) * std

    def _kl_divergence(
        self,
        q_mean: torch.Tensor,
        q_log_var: torch.Tensor,
        p_mean: torch.Tensor,
        p_log_var: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence KL[q || p] for two diagonal Gaussians (analytic)."""
        kl = 0.5 * (
            p_log_var
            - q_log_var
            + (q_log_var.exp() + (q_mean - p_mean).pow(2)) / p_log_var.exp()
            - 1
        )
        return kl.sum(-1).mean()

    def _gaussian_nll(
        self,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian negative log-likelihood (mean-reduced over batch and dim)."""
        nll = 0.5 * (
            log_var
            + (target - mean).pow(2) / log_var.exp()
            + math.log(2 * math.pi)
        )
        return nll.sum(-1).mean()

    def _process_sequence(
        self,
        x: torch.Tensor,
        x_1: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run VRNN steps over the input sequence.

        Args:
            x: Observations ``(batch, seq_len, input_dim)``.
            x_1: Next-step observations for posterior (training only); if
                ``None`` the prior is used at every step.

        Returns:
            h_final: Final hidden state ``(batch, hidden_size)``.
            total_kl: Mean KL divergence accumulated over the sequence.
        """
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        total_kl = torch.tensor(0.0, device=self.device)

        for t in range(seq_len):
            h_t = h.squeeze(0)  # (batch, hidden_size)

            # Prior distribution p(z_t | h_{t-1})
            p_h = self.prior_net(h_t)
            p_mean = self.prior_mean(p_h)
            p_log_var = self.prior_log_var(p_h)

            phi_xt = self.phi_x(x[:, t, :])  # (batch, hidden_size)

            # Posterior q(z_t | x_1_t, h_{t-1}) when x_1 is provided
            if x_1 is not None and t < x_1.size(1):
                enc_input = torch.cat([self.phi_x(x_1[:, t, :]), h_t], dim=-1)
            else:
                enc_input = torch.cat([phi_xt, h_t], dim=-1)

            enc_h = self.enc_net(enc_input)
            q_mean = self.enc_mean(enc_h)
            q_log_var = self.enc_log_var(enc_h)

            z_t = self._reparameterize(q_mean, q_log_var)
            phi_zt = self.phi_z(z_t)

            total_kl = total_kl + self._kl_divergence(
                q_mean, q_log_var, p_mean, p_log_var
            )

            rnn_input = torch.cat([phi_xt, phi_zt], dim=-1).unsqueeze(1)
            _, h = self.rnn(rnn_input, h)

        return h.squeeze(0), total_kl / seq_len

    def _decode_next_state(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample z from the prior and decode to a next-state distribution.

        Args:
            h: Hidden state ``(batch, hidden_size)``.

        Returns:
            z: Sampled latent vector ``(batch, latent_dim)``.
            y_mean: Predicted next-state mean ``(batch, output_dim)``.
            y_log_var: Predicted next-state log variance ``(batch, output_dim)``.
        """
        p_h = self.prior_net(h)
        p_mean = self.prior_mean(p_h)
        p_log_var = self.prior_log_var(p_h)
        z = self._reparameterize(p_mean, p_log_var)
        phi_z = self.phi_z(z)

        dec_input = torch.cat([phi_z, h], dim=-1)
        dec_h = self.dec_net(dec_input)
        return z, self.dec_mean(dec_h), self.dec_log_var(dec_h)
