"""Temporal Deep Latent Gaussian Model (tDLGM) implementation."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain

import torch
import torch.nn as nn


class TimeLayer(nn.Module):
    """Single LSTM layer that maps a time-series input to a hidden state.

    Args:
        input_dim: Number of input features per time step.
        hidden_size: LSTM hidden state size.
        device: PyTorch device.
    """

    def __init__(self, input_dim: int = 1, hidden_size: int = 1, device=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            device=self.device,
        )

    def forward(self, x: torch.Tensor):
        _, h = self.lstm(x)
        return h


class TimeRecognition(nn.Module):
    """Stack of :class:`TimeLayer` modules — one per generator layer.

    Args:
        input_dim: Number of input features per time step.
        hidden_size: LSTM hidden state size.
        seq_len: Length of the input sequence (informational).
        layers: Number of :class:`TimeLayer` instances.
        device: PyTorch device.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 1,
        seq_len: int = 1,
        layers: int = 1,
        device=None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.layers = layers
        self.device = device
        self.input_dim = input_dim

        self.time_layers = nn.ModuleList(
            TimeLayer(input_dim=input_dim, hidden_size=hidden_size, device=device)
            for _ in range(layers)
        )

    def forward(self, x: torch.Tensor):
        return [layer(x) for layer in self.time_layers]


class GenLayer(nn.Module):
    """One hierarchical layer of the tDLGM generator.

    Args:
        hidden_size: Size of the LSTM hidden / cell state.
        latent_dim: Dimension of the latent noise vector.
        seq_len: Sequence length (informational).
        device: PyTorch device.
    """

    def __init__(
        self,
        hidden_size: int = 1,
        latent_dim: int = 1,
        seq_len: int = 1,
        device=None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.device = device
        # None is intentional: call make_internal_state() to use explicit zeros.
        self.internal_state = None

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            device=self.device,
        )

        self.g = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, device=self.device),
            nn.Linear(self.latent_dim, self.hidden_size, device=self.device),
            nn.LeakyReLU(),
        )

    def get_internal_state(self):
        return self.internal_state

    def set_internal_state(self, internal_state) -> None:
        self.internal_state = internal_state

    def make_internal_state(self, batch_size: int = 1) -> None:
        self.internal_state = (
            torch.zeros(1, batch_size, self.hidden_size, device=self.device),
            torch.zeros(1, batch_size, self.hidden_size, device=self.device),
        )

    def forward(self, h: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        h, self.internal_state = self.lstm(h, self.internal_state)
        return h + self.g(xi)


class Generator(nn.Module):
    """Hierarchical generative network for tDLGM.

    Args:
        hidden_size: LSTM hidden state size.
        latent_dim: Dimension of each latent noise vector.
        output_dim: Dimension of the reconstructed output.
        layers: Number of :class:`GenLayer` instances.
        seq_len: Sequence length used when sampling noise.
        device: PyTorch device.
    """

    def __init__(
        self,
        hidden_size: int = 1,
        latent_dim: int = 1,
        output_dim: int = 1,
        layers: int = 1,
        seq_len: int = 1,
        device=None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.layers = layers
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.device = device
        self.xi = None

        self.gen_layers = nn.ModuleList(
            GenLayer(hidden_size, latent_dim, seq_len, device) for _ in range(layers)
        )

        self.initial_transform = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, device=device),
            nn.Linear(latent_dim, hidden_size, device=device),
            nn.Tanh(),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, output_dim, device=device),
            nn.Sigmoid(),
        )

    def forward(self, batch_size: int = 1):
        if self.xi is None:
            self.make_xi(batch_size)

        v = self.initial_transform(self.xi[0])
        for i, layer in enumerate(self.gen_layers, start=1):
            v = layer(v, self.xi[i])
        return self.output_layer(v[:, -1, :]), self.get_internal_state()

    def get_internal_state(self):
        return [layer.get_internal_state() for layer in self.gen_layers]

    def set_internal_state(self, internal_state) -> None:
        for layer, state in zip(self.gen_layers, internal_state, strict=False):
            layer.set_internal_state(state)

    def make_internal_state(self, batch_size: int = 1) -> None:
        for layer in self.gen_layers:
            layer.make_internal_state(batch_size)

    def set_xi(self, xi) -> None:
        self.xi = xi

    def make_xi(self, batch_size: int = 1) -> None:
        self.xi = [
            torch.randn(batch_size, self.seq_len, self.latent_dim, device=self.device)
            for _ in range(self.layers + 1)
        ]


class RecLayer(nn.Module):
    """Amortised recognition (encoder) layer that parameterises a Gaussian.

    Args:
        input_dim: Input feature dimension.
        latent_dim: Latent variable dimension.
        device: PyTorch device.
    """

    def __init__(self, input_dim: int = 1, latent_dim: int = 1, device=None) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device

        self.d = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Sigmoid(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        ).to(device)
        self.u = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Sigmoid(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        ).to(device)
        self.mean = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
        ).to(device)

    def forward(self, x: torch.Tensor):
        d = self.d(x)
        u = self.u(x)
        mean = self.mean(x)
        R = self._calculate_r(d, u)
        z = self._calculate_z(mean, R)
        return mean, R, z

    def _calculate_z(self, mean: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        v = torch.randn(*mean.size(), 1, device=self.device)
        mult = torch.matmul(R, v).squeeze(-1)
        return mult + mean

    def _calculate_r(self, d: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-6
        # D is diagonal, so its inverse is the elementwise reciprocal of d.
        # Clamp prevents inf when sigmoid output underflows to 0.
        d_safe = d.clamp(min=epsilon)
        D_inv = torch.diag_embed(1.0 / d_safe) + epsilon
        D_inv_sqrt = torch.sqrt(D_inv)
        u_r = u.unsqueeze(-1)
        U = torch.matmul(u_r, u_r.transpose(-2, -1))
        ut_d_inv_u = torch.matmul(u_r.transpose(-2, -1), torch.matmul(D_inv, u_r))
        eta = 1.0 / (1.0 + ut_d_inv_u)
        right = (1.0 - torch.sqrt(eta)) / (ut_d_inv_u + epsilon)
        return D_inv_sqrt - right * torch.matmul(D_inv, torch.matmul(U, D_inv_sqrt))


class Recognition(nn.Module):
    """Stack of :class:`RecLayer` modules — one per generator layer plus one.

    Args:
        input_dim: Input feature dimension.
        latent_dim: Latent variable dimension.
        layers: Number of generator layers (creates ``layers + 1`` rec layers).
        device: PyTorch device.
    """

    def __init__(
        self, input_dim: int = 1, latent_dim: int = 1, layers: int = 1, device=None
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers = layers
        self.device = device

        self.rec_layers = nn.ModuleList(
            RecLayer(input_dim, latent_dim, device) for _ in range(layers + 1)
        )

    def forward(self, x: torch.Tensor):
        means, Rs, zs = [], [], []
        for layer in self.rec_layers:
            mean, R, z = layer(x)
            means.append(mean)
            Rs.append(R)
            zs.append(z)
        return means, Rs, zs


class tDLGM(nn.Module):
    """Temporal Deep Latent Gaussian Model.

    Combines a time-recognition LSTM, a hierarchical latent generator, and an
    amortised recognition network to model temporal sequences.

    Args:
        input_dim: Feature dimension of each observed time step.
        hidden_size: LSTM hidden state size used by :class:`TimeRecognition`
            and :class:`Generator`.
        latent_dim: Dimension of each latent noise / code vector.
        output_dim: Dimension of the reconstructed output.
        layers: Number of hierarchical layers.
        seq_len: Length of the input sequence.
        device: PyTorch device.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 1,
        latent_dim: int = 1,
        output_dim: int = 1,
        layers: int = 1,
        seq_len: int = 1,
        device=None,
    ) -> None:
        super().__init__()

        self.model_t = TimeRecognition(input_dim, hidden_size, seq_len, layers, device)
        self.model_g = Generator(hidden_size, latent_dim, output_dim, layers, seq_len, device)
        self.model_r = Recognition(input_dim, latent_dim, layers, device)

        self.mse = nn.MSELoss()

    def get_parameters(self) -> Iterator[nn.Parameter]:
        """Return an iterator over all trainable parameters."""
        return chain(
            self.model_t.parameters(),
            self.model_g.parameters(),
            self.model_r.parameters(),
        )

    def _kl_reg_loss(
        self,
        mean: list,
        R: list,
        s: list,
        t_1: list,
        reg: float,
    ) -> torch.Tensor:
        matrix_size = mean[0].size(0) * mean[0].size(1)
        kl = torch.tensor(0.0, device=mean[0].device)
        for m, r in zip(mean, R, strict=False):
            C = r @ r.transpose(-2, -1)
            kl = kl + (
                0.5
                * torch.sum(
                    m.pow(2).sum(-1)
                    + C.diagonal(dim1=-2, dim2=-1).sum(-1)
                    - C.det().log()
                    - 1
                )
                / matrix_size
            )
        amount = len(s) * len(s[0])
        for a, b in zip(s, t_1, strict=False):
            kl = kl + reg * (self.mse(a[0], b[0]) + self.mse(a[1], b[1])) / amount
        return kl

    def _loss(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mean: list,
        R: list,
        s: list,
        t_1: list,
        reg: float,
    ) -> torch.Tensor:
        target = y.reshape_as(y_hat)
        return self.mse(y_hat, target) + self._kl_reg_loss(mean, R, s, t_1, reg)

    def get_loss(self, x: torch.Tensor, x_1: torch.Tensor, y: torch.Tensor) -> float:
        """Compute the loss without updating parameters.

        Args:
            x: Current-step observations ``(batch, seq_len, input_dim)``.
            x_1: Next-step observations ``(batch, seq_len, input_dim)``.
            y: Reconstruction targets ``(batch, 1, output_dim)``.

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
        """Perform a single training step (or evaluation step if optimizer is None).

        Args:
            x: Current-step observations ``(batch, seq_len, input_dim)``.
            x_1: Next-step observations ``(batch, seq_len, input_dim)``.
            y: Reconstruction targets ``(batch, 1, output_dim)``.
            optimizer: A PyTorch optimiser, or ``None`` for evaluation only.

        Returns:
            Scalar loss value.
        """
        if optimizer is not None:
            optimizer.zero_grad()

        t = self.model_t(x)
        t_1 = self.model_t(x_1)
        self.model_g.make_internal_state(x.size(0))
        self.model_g.set_internal_state(t)
        mean, R, z = self.model_r(x_1)
        self.model_g.set_xi(z)

        pred, h = self.model_g(x.size(0))

        loss = self._loss(y, pred, mean, R, h, t_1, reg=0.01)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
        return loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a prediction from observation ``x``.

        Args:
            x: Observations ``(batch, seq_len, input_dim)``.

        Returns:
            Predicted output tensor ``(batch, output_dim)``.
        """
        self.model_g.make_internal_state(x.size(0))
        t = self.model_t(x)
        self.model_g.set_internal_state(t)
        self.model_g.make_xi(x.size(0))
        val, _ = self.model_g(x.size(0))
        return val


class tDLGMCrossEntropy(tDLGM):
    """tDLGM variant that uses cross-entropy reconstruction loss.

    Suitable for classification targets (class indices or one-hot vectors).

    Args:
        input_dim: Feature dimension of each observed time step.
        hidden_size: LSTM hidden state size.
        latent_dim: Dimension of each latent noise / code vector.
        output_dim: Number of output classes.
        layers: Number of hierarchical layers.
        seq_len: Length of the input sequence.
        device: PyTorch device.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 1,
        latent_dim: int = 1,
        output_dim: int = 1,
        layers: int = 1,
        seq_len: int = 1,
        device=None,
    ) -> None:
        super().__init__(
            input_dim, hidden_size, latent_dim, output_dim, layers, seq_len, device
        )
        self.cross_entropy = nn.CrossEntropyLoss()

    def _loss(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mean: list,
        R: list,
        s: list,
        t_1: list,
        reg: float,
    ) -> torch.Tensor:
        # Accept either class indices or one-hot/logit vectors.
        if y.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            target = y.reshape(-1)
        else:
            target = y.argmax(dim=-1).reshape(-1)
        return self.cross_entropy(y_hat, target) + self._kl_reg_loss(mean, R, s, t_1, reg)
