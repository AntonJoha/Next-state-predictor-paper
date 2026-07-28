from collections.abc import Iterator
from itertools import chain

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Time Recognition ──────────────────────────────────────────────────────────


class TimeLayer(nn.Module):
    def __init__(self, input_dim=1, hidden_size=1, device=None):
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

    def forward(self, x):
        # No explicit initial state: PyTorch defaults to zeros, which is correct here.
        _, h = self.lstm(x)
        return h


class TimeRecognition(nn.Module):
    def __init__(self, input_dim=1, hidden_size=1, seq_len=1, layers=1, device=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.layers = layers
        self.device = device
        self.input_dim = input_dim

        self.time_layers = nn.ModuleList(
            TimeLayer(
                input_dim=input_dim,
                hidden_size=hidden_size,
                device=device,
            )
            for _ in range(layers)
        )

    def forward(self, x):
        return [layer(x) for layer in self.time_layers]


# ── Generator ─────────────────────────────────────────────────────────────────


class GenLayer(nn.Module):
    def __init__(self, hidden_size=1, latent_dim=1, seq_len=1, device=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.device = device
        # None is intentional: PyTorch LSTM accepts None as initial state (defaults to zeros).
        # Call make_internal_state() before forward() to use an explicit zero state.
        self.internal_state = None

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            device=self.device,
        )

        # Transform latent noise into hidden space
        self.g = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, device=self.device),
            nn.Linear(self.latent_dim, self.hidden_size, device=self.device),
            nn.LeakyReLU(),
        )

    def get_internal_state(self):
        return self.internal_state

    def set_internal_state(self, internal_state):
        self.internal_state = internal_state

    def make_internal_state(self, batch_size=1):
        self.internal_state = (
            torch.zeros(1, batch_size, self.hidden_size, device=self.device),
            torch.zeros(1, batch_size, self.hidden_size, device=self.device),
        )

    def forward(self, h, xi):
        h, self.internal_state = self.lstm(h, self.internal_state)
        return h + self.g(xi)


class Generator(nn.Module):
    def __init__(
        self,
        hidden_size=1,
        latent_dim=1,
        output_dim=1,
        layers=1,
        seq_len=1,
        device=None,
    ):
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

    def forward(self, batch_size=1):
        if self.xi is None:
            self.make_xi(batch_size)

        v = self.initial_transform(self.xi[0])
        for i, layer in enumerate(self.gen_layers, start=1):
            v = layer(v, self.xi[i])
        return self.output_layer(v[:, -1, :]), self.get_internal_state()

    def get_internal_state(self):
        return [layer.get_internal_state() for layer in self.gen_layers]

    def set_internal_state(self, internal_state):
        for layer, state in zip(self.gen_layers, internal_state, strict=False):
            layer.set_internal_state(state)

    def make_internal_state(self, batch_size=1):
        for layer in self.gen_layers:
            layer.make_internal_state(batch_size)

    def set_xi(self, xi):
        self.xi = xi

    def make_xi(self, batch_size=1):
        self.xi = [
            torch.randn(batch_size, self.seq_len, self.latent_dim, device=self.device)
            for _ in range(self.layers + 1)
        ]


# ── Recognition ───────────────────────────────────────────────────────────────


class RecLayer(nn.Module):
    def __init__(self, input_dim=1, latent_dim=1, device=None):
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

    def forward(self, x):
        d = self.d(x)
        u = self.u(x)
        mean = self.mean(x)
        R = self._calculate_r(d, u)
        z = self._calculate_z(mean, R)
        return mean, R, z

    def _calculate_z(self, mean, R):
        v = torch.randn(*mean.size(), 1, device=self.device)
        mult = torch.matmul(R, v).squeeze(-1)
        return mult + mean

    def _calculate_r(self, d, u):
        epsilon = 1e-6
        # D is diagonal, so its inverse is just the elementwise reciprocal of d.
        # Clamping d prevents inf when sigmoid output underflows to 0.
        d_safe = d.clamp(min=epsilon)
        # Adding epsilon after diag_embed ensures every element (including
        # off-diagonals) is positive before sqrt, avoiding undefined gradients.
        D_inv = torch.diag_embed(1.0 / d_safe) + epsilon
        D_inv_sqrt = torch.sqrt(D_inv)
        u_r = u.unsqueeze(-1)
        U = torch.matmul(u_r, u_r.transpose(-2, -1))
        ut_d_inv_u = torch.matmul(u_r.transpose(-2, -1), torch.matmul(D_inv, u_r))
        eta = 1.0 / (1.0 + ut_d_inv_u)
        # Add epsilon to denominator to guard against ut_d_inv_u ≈ 0.
        right = (1.0 - torch.sqrt(eta)) / (ut_d_inv_u + epsilon)
        return D_inv_sqrt - right * torch.matmul(D_inv, torch.matmul(U, D_inv_sqrt))


class Recognition(nn.Module):
    def __init__(self, input_dim=1, latent_dim=1, layers=1, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layers = layers
        self.device = device

        self.rec_layers = nn.ModuleList(
            RecLayer(input_dim, latent_dim, device) for _ in range(layers + 1)
        )

    def forward(self, x):
        means, Rs, zs = [], [], []
        for layer in self.rec_layers:
            mean, R, z = layer(x)
            means.append(mean)
            Rs.append(R)
            zs.append(z)
        return means, Rs, zs


# ── tDLGM ─────────────────────────────────────────────────────────────────────


class tDLGM(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_size=1,
        latent_dim=1,
        output_dim=1,
        layers=1,
        seq_len=1,
        device=None,
    ):
        super().__init__()

        self.model_t = TimeRecognition(input_dim,
                                       hidden_size,
                                       seq_len,
                                       layers,
                                       device)

        self.model_g = Generator(hidden_size,
                                 latent_dim,
                                 output_dim,
                                 layers,
                                 seq_len,
                                 device)

        self.model_r = Recognition(input_dim, latent_dim, layers, device)

        self.mse = nn.MSELoss()

    def get_parameters(self) -> Iterator[nn.Parameter]:
        return chain(
            self.model_t.parameters(),
            self.model_g.parameters(),
            self.model_r.parameters(),
        )

    def _loss(self, y, y_hat, mean, R, s, t_1, reg) -> torch.Tensor:
        target = y.reshape_as(y_hat)
        loss = self.mse(y_hat, target)
        matrix_size = mean[0].size(0) * mean[0].size(1)

        for m, r in zip(mean, R, strict=False):
            C = r @ r.transpose(-2, -1)
            det = C.det()
            loss += (
                0.5
                * torch.sum(
                    m.pow(2).sum(-1)
                    + C.diagonal(dim1=-2, dim2=-1).sum(-1)
                    - det.log()
                    - 1
                )
                / matrix_size
            )

        amount = len(s) * len(s[0])
        for a, b in zip(s, t_1, strict=False):
            loss += reg * (self.mse(a[0], b[0]) + self.mse(a[1], b[1])) / amount

        return loss

    def get_loss(self, x, x_1, y) -> float:
        return self.train_step(x, x_1, y, optimizer=None)

    def train_step(self, x, x_1, y, optimizer) -> float:
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

    def forward(self, x) -> torch.Tensor:
        self.model_g.make_internal_state(x.size(0))
        t = self.model_t(x)
        self.model_g.set_internal_state(t)
        self.model_g.make_xi(x.size(0))
        val, _ = self.model_g(x.size(0))
        return val


class tDLGMCrossEntropy(tDLGM):
    def __init__(
        self,
        input_dim=1,
        hidden_size=1,
        latent_dim=1,
        output_dim=1,
        layers=1,
        seq_len=1,
        device=None,
    ):
        super().__init__(
            input_dim, hidden_size, latent_dim, output_dim, layers, seq_len, device
        )
        self.cross_entropy = nn.CrossEntropyLoss()

    def _loss(self, y, y_hat, mean, R, s, t_1, reg) -> torch.Tensor:
        target = y.argmax(dim=-1)
        loss = self.cross_entropy(y_hat, target)
        matrix_size = mean[0].size(0) * mean[0].size(1)

        for m, r in zip(mean, R, strict=False):
            C = r @ r.transpose(-2, -1)
            det = C.det()
            loss += (
                0.5
                * torch.sum(
                    m.pow(2).sum(-1)
                    + C.diagonal(dim1=-2, dim2=-1).sum(-1)
                    - det.log()
                    - 1
                )
                / matrix_size
            )

        amount = len(s) * len(s[0])
        for a, b in zip(s, t_1, strict=False):
            loss += reg * (self.mse(a[0], b[0]) + self.mse(a[1], b[1])) / amount

        return loss


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch.optim import Adam

    model = tDLGM(
        input_dim=10,
        hidden_size=20,
        latent_dim=5,
        output_dim=10,
        layers=2,
        seq_len=3,
        device=device,
    ).to(device)
    optimizer = Adam(model.get_parameters(), lr=0.1)

    x = torch.randn(50, 3, 10).to(device)  # used for the state recognition
    y = torch.randn(50, 1, 10).to(device)  # the value to be reconstructed
    x_1 = torch.cat((x, y), dim=1)[:, 1:, :]  # used for the recognition

    before = model.get_loss(x, x_1, y)
    for _ in range(300):
        loss = model.train_step(x, x_1, y, optimizer)
    after = model.get_loss(x, x_1, y)
    print(f"Loss before training: {before}")
    print(f"Loss after training: {after}")
    assert after < before, "Loss did not decrease after training! MSELoss"

    model = tDLGMCrossEntropy(
        input_dim=10,
        hidden_size=20,
        latent_dim=5,
        output_dim=10,
        layers=2,
        seq_len=3,
        device=device,
    ).to(device)
    optimizer = Adam(model.get_parameters(), lr=0.1)

    x = torch.randn(50, 3, 10).to(device)  # used for the state recognition
    y = torch.randint(0, 10, (50, 1)).to(device)  # the value to be reconstructed (class labels)
    x_1 = torch.cat((x, nn.functional.one_hot(y.squeeze(), num_classes=10).float().unsqueeze(1)), dim=1)[:, 1:, :]  # used for the recognition

    before = model.get_loss(x, x_1, y)
    for _ in range(300):
        loss = model.train_step(x, x_1, y, optimizer)
    after = model.get_loss(x, x_1, y)
    print(f"Loss before training: {before}")
    print(f"Loss after training: {after}")
    assert after < before, "Loss did not decrease after training! CrossEntropyLoss"
