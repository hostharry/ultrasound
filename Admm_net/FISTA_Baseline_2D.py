"""2D 对比模型: HASA-FISTA (参考 HASA_FISTA_Net_3.py).

设计要点:
  - Data consistency: z = v - rho * A^T(A(v)-y)
  - HASA 权重: 逐像素 lambda_tv / lambda_wav / alpha
  - 双分支 prox: TV branch + WAV branch
  - FISTA momentum: v_next = x_next + beta * (x_next - x_prev)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from admm_ops import soft_threshold


class HASAWeightFISTA2D(nn.Module):
    """HASA-FISTA 专用权重网络: 输出 lambda_tv/lambda_wav/alpha."""

    def __init__(self, hidden_ch=16, num_layers=2, inner_ks=5):
        super().__init__()
        layers = [nn.Conv2d(1, hidden_ch, 3, padding=1), nn.ReLU(inplace=True)]
        inner_pad = inner_ks // 2
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(hidden_ch, hidden_ch, inner_ks, padding=inner_pad),
                nn.ReLU(inplace=True),
            ])
        self.feat_net = nn.Sequential(*layers)
        self.head_tv = nn.Sequential(nn.Conv2d(hidden_ch, 1, 1), nn.Softplus())
        self.head_wav = nn.Sequential(nn.Conv2d(hidden_ch, 1, 1), nn.Softplus())
        self.head_alpha = nn.Sequential(nn.Conv2d(hidden_ch, 1, 1), nn.Sigmoid())

        nn.init.constant_(self.head_tv[0].bias, -4.0)
        nn.init.constant_(self.head_wav[0].bias, -4.0)
        nn.init.constant_(self.head_alpha[0].bias, 0.0)

    def forward(self, x):
        feat = self.feat_net(x)
        return self.head_tv(feat), self.head_wav(feat), self.head_alpha(feat)


class ISTAProx2D_Dual(nn.Module):
    """2D 双分支 ISTA Prox.

    TV branch:  E_tv -> shrink(thr_tv) -> D_tv
    WAV branch: E_wav -> shrink(thr_wav) -> D_wav
    """

    def __init__(self, feat_ch=32, k=3):
        super().__init__()
        pad = k // 2

        self.E1_tv = nn.Conv2d(1, feat_ch, k, padding=pad)
        self.E2_tv = nn.Conv2d(feat_ch, feat_ch, k, padding=pad)
        self.D1_tv = nn.Conv2d(feat_ch, feat_ch, k, padding=pad)
        self.D2_tv = nn.Conv2d(feat_ch, 1, k, padding=pad)

        self.E1_wav = nn.Conv2d(1, feat_ch, k, padding=pad)
        self.E2_wav = nn.Conv2d(feat_ch, feat_ch, k, padding=pad)
        self.D1_wav = nn.Conv2d(feat_ch, feat_ch, k, padding=pad)
        self.D2_wav = nn.Conv2d(feat_ch, 1, k, padding=pad)

    @staticmethod
    def _expand_thr(thr, feat):
        # thr: (B,1,H,W), feat: (B,C,H,W)
        return thr.expand(-1, feat.shape[1], -1, -1)

    def _branch_forward(self, z, thr, e1, e2, d1, d2, return_symloss=False):
        x = F.relu(e1(z))
        feat = e2(x)
        feat_shrink = soft_threshold(feat, self._expand_thr(thr, feat))
        x_next = d2(F.relu(d1(feat_shrink)))

        if not return_symloss:
            return x_next, None

        x_est = d2(F.relu(d1(feat)))
        symloss = x_est - z
        return x_next, symloss

    def forward(self, z, thr_tv, thr_wav, alpha=None, return_symloss=False, return_branches=False):
        x_tv, sym_tv = self._branch_forward(
            z, thr_tv, self.E1_tv, self.E2_tv, self.D1_tv, self.D2_tv, return_symloss
        )
        x_wav, sym_wav = self._branch_forward(
            z, thr_wav, self.E1_wav, self.E2_wav, self.D1_wav, self.D2_wav, return_symloss
        )

        if alpha is None:
            x_next = 0.5 * (x_tv + x_wav)
        else:
            x_next = alpha * x_tv + (1.0 - alpha) * x_wav

        if not return_branches:
            return x_next, sym_tv, sym_wav, None, None
        return x_next, sym_tv, sym_wav, x_tv, x_wav


class HASA_FISTA_Block_2D(nn.Module):
    """单层 HASA-FISTA block."""

    def __init__(self, weight_ctor, feat_ch=32, prox_k=3):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.weight = weight_ctor()
        self.prox = ISTAProx2D_Dual(feat_ch=feat_ch, k=prox_k)

    def forward(self, x_prev, v, y, op, return_symloss=False, return_branches=False):
        # data consistency: z = v - rho * A^T(A(v)-y)
        r = op.A(v) - y
        g = op.At(r)
        z = v - F.softplus(self.rho) * g

        lambda_tv, lambda_wav, alpha = self.weight(z)
        thr_tv = F.softplus(self.soft_thr) * lambda_tv
        thr_wav = F.softplus(self.soft_thr) * lambda_wav

        x_next, sym_tv, sym_wav, x_tv, x_wav = self.prox(
            z,
            thr_tv,
            thr_wav,
            alpha=alpha,
            return_symloss=return_symloss,
            return_branches=return_branches,
        )

        beta = torch.tanh(self.beta)
        v_next = x_next + beta * (x_next - x_prev)

        # 为了兼容 CombinedLoss 的 constraint 键, 返回零占位.
        zeros = torch.zeros_like(x_next)
        aux = {
            "rho1": F.softplus(self.rho).detach(),  # 与 ADMM 日志字段对齐
            "rho2": F.softplus(self.rho).detach(),
            "eta": F.softplus(self.soft_thr).detach(),
            "gamma": torch.tensor(0.0, device=x_next.device),
            "lambda_tv": lambda_tv.detach(),
            "lambda_wav": lambda_wav.detach(),
            "alpha": alpha.detach(),
            "sym_tv": sym_tv,
            "sym_wav": sym_wav,
            "constraint_wav": zeros.detach(),
            "constraint_tv": zeros.detach(),
        }
        if return_branches:
            aux["x_tv"] = x_tv
            aux["x_wav"] = x_wav
        return x_next, v_next, aux


class HASA_FISTA_Net_2D(nn.Module):
    """2D HASA-FISTA 网络."""

    def __init__(self, layer_num=12, hasa_ctor=None, feat_ch=32, prox_k=3):
        super().__init__()
        if hasa_ctor is None:
            hasa_ctor = lambda: HASAWeightFISTA2D()

        self.blocks = nn.ModuleList(
            [HASA_FISTA_Block_2D(hasa_ctor, feat_ch=feat_ch, prox_k=prox_k) for _ in range(layer_num)]
        )

    def forward(self, y, op, x0=None, return_aux=False, return_symloss=False, return_branches=False):
        if x0 is None:
            x0 = op.At(y)

        scale = x0.abs().amax(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-6)
        x = x0 / scale
        v = x0 / scale
        y_scaled = y / scale.view(scale.shape[0], 1, 1)

        aux_list = []
        for blk in self.blocks:
            x, v, aux = blk(
                x,
                v,
                y_scaled,
                op,
                return_symloss=return_symloss,
                return_branches=return_branches,
            )
            if return_aux or return_symloss or return_branches:
                aux_list.append(aux)

        x = x * scale

        if return_aux or return_symloss or return_branches:
            return x, aux_list
        return x
