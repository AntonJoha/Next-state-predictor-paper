"""Denoising Diffusion Probabilistic Model (DDPM) for next-state prediction."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timesteps.

    Args:
        t: Integer timestep tensor ``(batch,)``.
        dim: Embedding dimension (must be even).

    Returns:
        Embedding tensor ``(batch, dim)``.
    """
    half = dim // 2
    denom = max(half - 1, 1)
    freqs = torch.exp(
        -math.log(10_000)
        * torch.arange(half, device=t.device, dtype=torch.float)
        / denom
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _NoisePredictor(nn.Module):
    """MLP that estimates the noise added at a given diffusion step.

    Takes the noisy target, a sinusoidal timestep embedding, and a context
    vector as input, and returns the predicted noise with the same shape as
    the target.

    Args:
        output_dim: Dimension of the target (next state).
        hidden_size: Width of the hidden layers.
        t_emb_dim: Dimension of the timestep embedding.
        device: PyTorch device.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_size: int,
        t_emb_dim: int,
        device=None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(output_dim + t_emb_dim + hidden_size, hidden_size * 2, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, output_dim, device=device),
        )

    def forward(
        self,
        y_t: torch.Tensor,
        t_emb: torch.Tensor,
        ctx: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([y_t, t_emb, ctx], dim=-1))


class DiffusionPredictor(nn.Module):
    """Conditional DDPM next-state predictor.

    A GRU sequence encoder compresses the current observation window into a
    context vector.  A noise-prediction MLP conditioned on this context is
    then trained with the standard DDPM denoising objective, which is a
    variational lower bound on the negative log-likelihood of the target
    next state.

    At inference time the model runs the full reverse diffusion chain
    starting from Gaussian noise, conditioned on the encoded observations,
    to produce a point-estimate prediction (the mean of the terminal
    denoised distribution).

    Args:
        input_dim: Feature dimension of each observed time step.
        hidden_size: GRU encoder size and MLP hidden width.
        output_dim: Dimension of the predicted next state.
        T: Number of diffusion steps (for both training and inference).
        t_emb_dim: Dimension of the sinusoidal timestep embedding (must be even).
        beta_start: Smallest value in the linear noise schedule.
        beta_end: Largest value in the linear noise schedule.
        device: PyTorch device.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 64,
        output_dim: int = 1,
        T: int = 200,
        t_emb_dim: int = 16,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device=None,
    ) -> None:
        super().__init__()
        if t_emb_dim % 2 != 0:
            msg = "t_emb_dim must be even for sinusoidal embedding."
            raise ValueError(msg)

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.T = T
        self.t_emb_dim = t_emb_dim
        self.device = device

        self.context_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            device=device,
        )

        self.noise_predictor = _NoisePredictor(
            output_dim=output_dim,
            hidden_size=hidden_size,
            t_emb_dim=t_emb_dim,
            device=device,
        )

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def get_parameters(self) -> Iterator[nn.Parameter]:
        """Return an iterator over all trainable parameters."""
        return self.parameters()

    def get_loss(self, x: torch.Tensor, x_1: torch.Tensor, y: torch.Tensor) -> float:
        """Compute the DDPM denoising loss without updating parameters.

        Args:
            x: Current-step observations ``(batch, seq_len, input_dim)``.
            x_1: Next-step observations (unused; present for API compatibility).
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

        The DDPM denoising objective is a variational lower bound on the
        negative log-likelihood of ``y`` given ``x``:

        .. math::
            \\mathcal{L}_{\\text{DDPM}} =
                \\mathbb{E}_{t, \\varepsilon}
                \\left[\\|\\varepsilon - \\varepsilon_\\theta(
                    \\sqrt{\\bar\\alpha_t}\\,y + \\sqrt{1-\\bar\\alpha_t}\\,
                    \\varepsilon,\\; t,\\; \\text{ctx})\\|^2\\right]

        Args:
            x: Current-step observations ``(batch, seq_len, input_dim)``.
            x_1: Next-step observations (unused; present for API compatibility).
            y: Target next states ``(batch, 1, output_dim)`` or
                ``(batch, output_dim)``.
            optimizer: A PyTorch optimiser, or ``None`` for evaluation only.

        Returns:
            Scalar loss value.
        """
        if optimizer is not None:
            optimizer.zero_grad()

        context = self._encode_context(x)
        y_0 = y.reshape(x.size(0), -1)[:, : self.output_dim]
        loss = self._ddpm_loss(y_0, context)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
        return loss.item()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a next-state prediction via the reverse diffusion process.

        Starts from standard Gaussian noise and iteratively denoises,
        conditioned on the encoded observation sequence.

        Args:
            x: Observations ``(batch, seq_len, input_dim)``.

        Returns:
            Predicted next state ``(batch, output_dim)``.
        """
        batch_size = x.size(0)
        context = self._encode_context(x)
        y = torch.randn(batch_size, self.output_dim, device=self.device)

        for t_idx in reversed(range(self.T)):
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            t_emb = _sinusoidal_embedding(t, self.t_emb_dim)
            pred_noise = self.noise_predictor(y, t_emb, context)

            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alpha_bar[t_idx]
            beta_t = self.betas[t_idx]

            mean = (
                y - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise
            ) / torch.sqrt(alpha_t)

            if t_idx > 0:
                y = mean + torch.sqrt(beta_t) * torch.randn_like(y)
            else:
                y = mean

        return y

    def _encode_context(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the observation sequence into a fixed-size context vector."""
        _, h = self.context_encoder(x)
        return h.squeeze(0)

    def _q_sample(
        self,
        y_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """q(y_t | y_0) = N(sqrt(α̅_t)·y_0, (1−α̅_t)·I)."""
        alpha_bar_t = self.alpha_bar[t][:, None]
        return torch.sqrt(alpha_bar_t) * y_0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def _ddpm_loss(
        self, y_0: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Compute the DDPM noise-prediction MSE loss."""
        batch_size = y_0.size(0)
        t = torch.randint(0, self.T, (batch_size,), device=self.device)
        noise = torch.randn_like(y_0)
        y_t = self._q_sample(y_0, t, noise)
        t_emb = _sinusoidal_embedding(t, self.t_emb_dim)
        pred_noise = self.noise_predictor(y_t, t_emb, context)
        return F.mse_loss(pred_noise, noise)
