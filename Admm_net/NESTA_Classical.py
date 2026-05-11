"""Classical NESTA Algorithm  (Becker, Bobin & Candès, SIAM J. Imaging Sci. 2011)

Solves the Basis-Pursuit Denoising (BPDN) problem:

    min  ||W* x||_1   subject to   ||y_sub − A x||_2  ≤  η

via Nesterov's smoothing technique + accelerated projected gradient descent.

Implements **Algorithm 1** (plain NESTA) and **Algorithm 2** (restarted NESTA)
as described in the NESTANets paper (Neyra-Nesterenko & Adcock, 2022).

NO learnable parameters — pure iterative optimisation baseline.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


# ======================== Sparsifying transforms W ========================

class SparsifyBase(ABC):
    """Interface for analysis operator W used in  min ||W* x||_1."""

    @property
    @abstractmethod
    def beta(self) -> float:
        """Upper bound on ||W||^2 (Lipschitz constant of ∇f_µ)."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """W* x  (analysis: signal → sparse coefficients)."""

    @abstractmethod
    def adjoint(self, u: torch.Tensor) -> torch.Tensor:
        """W u   (synthesis: coefficients → signal)."""


class IdentityTransform(SparsifyBase):
    """W = I  — assume sparsity in the signal domain itself."""

    @property
    def beta(self) -> float:
        return 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def adjoint(self, u: torch.Tensor) -> torch.Tensor:
        return u


class FiniteDiffTransform(SparsifyBase):
    """W = D  (first-order finite differences) — Total-Variation prior.

    W* x  = x[..., 1:] − x[..., :-1]   (length N → N−1)
    W u   = D^T u                         (length N−1 → N)
    """

    @property
    def beta(self) -> float:
        return 4.0  # ||D||^2 ≤ 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:] - x[..., :-1]

    def adjoint(self, u: torch.Tensor) -> torch.Tensor:
        pad_l = -u[..., :1]
        pad_r = u[..., -1:]
        mid = u[..., :-1] - u[..., 1:]
        return torch.cat([pad_l, mid, pad_r], dim=-1)


class DCTTransform(SparsifyBase):
    """W = Type-II DCT (orthonormal) — frequency-domain sparsity.

    Requires PyTorch ≥ 2.1 for ``torch.fft.dct`` / ``torch.fft.idct``.
    Falls back to a manual implementation if unavailable.
    """

    @property
    def beta(self) -> float:
        return 1.0  # orthonormal ⇒ ||W||^2 = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(torch.fft, "dct"):
            return torch.fft.dct(x, norm="ortho")
        return self._dct_manual(x)

    def adjoint(self, u: torch.Tensor) -> torch.Tensor:
        if hasattr(torch.fft, "idct"):
            return torch.fft.idct(u, norm="ortho")
        return self._idct_manual(u)

    @staticmethod
    def _dct_manual(x: torch.Tensor) -> torch.Tensor:
        N = x.shape[-1]
        v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
        Vc = torch.fft.fft(v, dim=-1)
        k = torch.arange(N, device=x.device, dtype=x.dtype)
        W = torch.exp(-1j * math.pi * k / (2 * N))
        out = (Vc * W).real * math.sqrt(2.0 / N)
        out[..., 0] /= math.sqrt(2.0)
        return out

    @staticmethod
    def _idct_manual(X: torch.Tensor) -> torch.Tensor:
        N = X.shape[-1]
        Xc = X.clone()
        Xc[..., 0] *= math.sqrt(2.0)
        Xc = Xc * math.sqrt(N / 2.0)
        k = torch.arange(N, device=X.device, dtype=X.dtype)
        W = torch.exp(1j * math.pi * k / (2 * N))
        Vc = torch.complex(Xc, torch.zeros_like(Xc)) * W
        v = torch.fft.ifft(Vc, dim=-1).real
        out = torch.zeros_like(X)
        out[..., ::2] = v[..., : (N + 1) // 2]
        out[..., 1::2] = v[..., (N + 1) // 2 :].flip(-1)
        return out


_TRANSFORMS = {
    "identity": IdentityTransform,
    "tv": FiniteDiffTransform,
    "dct": DCTTransform,
}


def get_transform(name: str) -> SparsifyBase:
    if name not in _TRANSFORMS:
        raise ValueError(f"Unknown transform '{name}'. Choose from {list(_TRANSFORMS)}")
    return _TRANSFORMS[name]()


# ======================== Classical NESTA ========================

class ClassicalNESTA(nn.Module):
    """Classical NESTA (no learnable parameters).

    Parameters
    ----------
    n_iters : int
        Inner iterations per restart (or total if *n_restarts* = 0).
    mu : float
        Smoothing parameter for plain NESTA (ignored when *n_restarts* > 0).
    eta : float
        Noise tolerance  ||y − Ax||₂ ≤ η.  Set relative to data if unsure
        (e.g. 1e-4 × ||y||).
    n_restarts : int
        Number of restarts.  0 → plain NESTA (Algorithm 1 only).
    restart_decay : float
        Contraction rate *r* ∈ (0, 1) for restart scheme (default 0.25).
    zeta : float
        Target error floor for restart smoothing schedule (default 1e-9).
    transform : str
        Sparsifying transform W.  ``"identity"`` | ``"tv"`` | ``"dct"``.
    """

    def __init__(
        self,
        n_iters: int = 60,
        mu: float = 1e-3,
        eta: float = 1e-4,
        n_restarts: int = 0,
        restart_decay: float = 0.25,
        zeta: float = 1e-9,
        transform: str = "identity",
    ):
        super().__init__()
        self.n_iters = n_iters
        self.mu = mu
        self.eta = eta
        self.n_restarts = n_restarts
        self.restart_decay = restart_decay
        self.zeta = zeta
        self.W: SparsifyBase = get_transform(transform)

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _T_mu(x: torch.Tensor, mu: float) -> torch.Tensor:
        r"""Gradient of the Huber-smoothed :math:`\ell_1` norm.

        .. math::
            T_\mu(x_i) = \begin{cases}
                x_i / \mu  & |x_i| \le \mu \\
                \mathrm{sign}(x_i) & |x_i| > \mu
            \end{cases}
        """
        abs_x = torch.abs(x)
        return torch.where(abs_x <= mu, x / (mu + 1e-30), torch.sign(x))

    def _complex_norm(self, z: torch.Tensor) -> torch.Tensor:
        """Per-batch L2 norm that handles both real and complex tensors."""
        flat = z.reshape(z.shape[0], -1)
        if z.is_complex():
            return torch.sqrt((flat.real ** 2 + flat.imag ** 2).sum(dim=1))
        return torch.norm(flat, dim=1)

    def _project(
        self, q: torch.Tensor, y_sub: torch.Tensor, op, eta: float,
    ) -> torch.Tensor:
        r"""Project *q* onto the feasibility set :math:`Q = \{x : \|y - Ax\|_2 \le \eta\}`.

        Closed-form for tight frame  :math:`AA^* = \nu I`  with :math:`\nu = 1`:

        .. math::
            \lambda = \max(0,\; \|y - Aq\|_2 / \eta - 1)  \\
            P_Q(q) = q + \frac{\lambda}{\lambda + 1}\; A^*(y - Aq)
        """
        residual = y_sub - op.A(q)
        res_norm = self._complex_norm(residual)          # (B,)
        lam = torch.clamp(res_norm / eta - 1.0, min=0.0)  # (B,)
        scale = lam / (lam + 1.0)                          # (B,)

        correction = op.At(residual)                       # (B, 1, N)
        scale = scale.view(-1, *([1] * (correction.dim() - 1)))
        return q + scale * correction

    # ---- Algorithm 1 : NESTA inner loop ---------------------------------

    def _nesta_inner(
        self,
        y_sub: torch.Tensor,
        op,
        z0: torch.Tensor,
        mu: float,
        n_iters: int,
    ) -> torch.Tensor:
        """Run *n_iters* of NESTA (Algorithm 1) with fixed smoothing µ."""
        W = self.W
        beta = W.beta
        z = z0.clone()
        sum_alpha_grad = torch.zeros_like(z0)
        x_n = z0.clone()

        for n in range(n_iters):
            alpha_n = (n + 1) / 2.0
            tau_n = 2.0 / (n + 3)

            # ∇f_µ(z) = (µ/β) · W · T_µ(W* z)
            Wt_z = W.forward(z)
            grad_f = (mu / beta) * W.adjoint(self._T_mu(Wt_z, mu))

            # --- x_n : gradient step + projection ---
            q_x = z - grad_f
            x_n = self._project(q_x, y_sub, op, self.eta)

            # --- v_n : accumulated gradient + projection ---
            Wt_z_for_acc = Wt_z  # reuse W*z already computed
            sum_alpha_grad = sum_alpha_grad + alpha_n * W.adjoint(
                self._T_mu(Wt_z_for_acc, mu)
            )
            q_v = z0 - (mu / beta) * sum_alpha_grad
            v_n = self._project(q_v, y_sub, op, self.eta)

            # --- momentum update ---
            z = tau_n * v_n + (1.0 - tau_n) * x_n

        return x_n

    # ---- Algorithm 2 : Restarted NESTA ----------------------------------

    @torch.no_grad()
    def forward(self, y_sub, op, return_aux=False):
        x0 = op.At(y_sub)  # zero-filled initial reconstruction

        if self.n_restarts > 0:
            # --- Restarted NESTA (Algorithm 2) ---
            x_star = torch.zeros_like(x0)
            eps_k = torch.norm(
                x0.reshape(x0.shape[0], -1), dim=1
            ).mean().item()
            r = self.restart_decay

            aux_list = []
            for k in range(self.n_restarts):
                mu_k = max(eps_k, 1e-15)
                x_star = self._nesta_inner(y_sub, op, x_star, mu_k, self.n_iters)
                eps_k = r * eps_k + self.zeta

                if return_aux:
                    res = y_sub - op.A(x_star)
                    rn = self._complex_norm(res).mean().item()
                    l1 = torch.norm(
                        x_star.reshape(x_star.shape[0], -1), p=1, dim=1
                    ).mean().item()
                    aux_list.append({
                        "restart": k, "mu": mu_k,
                        "residual_norm": rn, "l1_objective": l1,
                    })
        else:
            # --- Plain NESTA (Algorithm 1 only) ---
            x_star = self._nesta_inner(y_sub, op, x0, self.mu, self.n_iters)
            aux_list = []

        if return_aux:
            return x_star, aux_list
        return x_star

    def extra_repr(self) -> str:
        return (
            f"n_iters={self.n_iters}, mu={self.mu}, eta={self.eta}, "
            f"n_restarts={self.n_restarts}, restart_decay={self.restart_decay}, "
            f"transform={type(self.W).__name__}"
        )
