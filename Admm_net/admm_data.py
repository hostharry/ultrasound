import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class MaskedRFFT1D:
    """1D 频域掩膜采样算子"""
    N: int
    mask: torch.Tensor

    def __post_init__(self):
        self.N = int(self.N)
        self.mask = self.mask.to(torch.bool)
        self.mu = torch.nonzero(self.mask, as_tuple=False).squeeze(1)

    @property
    def K(self) -> int:
        return int(self.mu.numel())

    def A(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.squeeze(1) if x.dim() == 3 else x
        return torch.fft.rfft(x_flat, dim=-1).index_select(dim=-1, index=self.mu)

    def At(self, y: torch.Tensor) -> torch.Tensor:
        B = y.shape[0]
        full = torch.zeros((B, self.N // 2 + 1), device=y.device, dtype=y.dtype)
        full.index_copy_(dim=-1, index=self.mu, source=y)
        x = torch.fft.irfft(full, n=self.N, dim=-1)
        return x.unsqueeze(1)


@dataclass
class MaskedRFFT2D:
    """2D 测量算子: 逐行施加 1D rfft + 共享掩膜

    与 MaskedRFFT1D 的数学完全相同, 但批量化到 H 行.
    所有行共享同一个频域掩膜 (超声阵列各阵元使用相同采样方案).
    """
    N: int
    mask: torch.Tensor

    def __post_init__(self):
        self.N = int(self.N)
        self.mask = self.mask.to(torch.bool)
        self.mu = torch.nonzero(self.mask, as_tuple=False).squeeze(1)

    @property
    def K(self) -> int:
        return int(self.mu.numel())

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) → (B, H, K) complex"""
        x_2d = x.squeeze(1)
        X_freq = torch.fft.rfft(x_2d, dim=-1)
        return X_freq.index_select(dim=-1, index=self.mu)

    def At(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, H, K) complex → (B, 1, H, W)"""
        B, H, _ = y.shape
        full = torch.zeros(B, H, self.N // 2 + 1, device=y.device, dtype=y.dtype)
        mu_exp = self.mu.unsqueeze(0).unsqueeze(0).expand(B, H, -1)
        full.scatter_(2, mu_exp, y)
        x = torch.fft.irfft(full, n=self.N, dim=-1)
        return x.unsqueeze(1)


class UltrasoundDataset:
    """从预处理 npz 文件加载超声数据集"""

    def __init__(self, npz_path: str, cs_ratio: int, device: str = "cpu"):
        data = np.load(npz_path, allow_pickle=True)
        self.Y = torch.from_numpy(data["Y"]).float().unsqueeze(1)
        self.X = torch.from_numpy(data[f"X{cs_ratio}"]).float().unsqueeze(1)
        self.Y_k = torch.from_numpy(data[f"Y{cs_ratio}_k"]) if f"Y{cs_ratio}_k" in data else None
        self.mask = torch.from_numpy(data[f"mask{cs_ratio}"])
        self.group_id = torch.from_numpy(data["group_id"]).long() if "group_id" in data else None
        self.source_id = torch.from_numpy(data["source_id"]).long() if "source_id" in data else None
        self.N = self.Y.shape[-1]
        self.fs = float(data["fs"])
        self.fc = float(data["fc"])
        self.op = MaskedRFFT1D(N=self.N, mask=self.mask.clone())
        self.device = device

    def to(self, device: str) -> "UltrasoundDataset":
        self.device = device
        self.Y, self.X = self.Y.to(device), self.X.to(device)
        if self.Y_k is not None:
            self.Y_k = self.Y_k.to(device)
        if self.group_id is not None:
            self.group_id = self.group_id.to(device)
        if self.source_id is not None:
            self.source_id = self.source_id.to(device)
        self.mask = self.mask.to(device)
        self.op = MaskedRFFT1D(N=self.N, mask=self.mask.clone())
        return self

    def get_batch(self, idx) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.X[idx], self.Y[idx], self.Y_k[idx] if self.Y_k is not None else None

    def __len__(self) -> int:
        return self.Y.shape[0]


class UltrasoundFrameDataset:
    """帧级超声数据集 (2D), 支持 patch 提取

    从 prepare_picmus_data.py --mode frame 生成的 npz 加载.
    每个样本是一个 (patch_h, W) 的 2D RF 帧片段.
    """

    def __init__(self, npz_path: str, cs_ratio: int,
                 patch_h: Optional[int] = None,
                 patch_stride: Optional[int] = None,
                 device: str = "cpu"):
        data = np.load(npz_path, allow_pickle=True)
        self.Y_frames = torch.from_numpy(data["Y_frames"]).float()
        self.X_frames = torch.from_numpy(data[f"X{cs_ratio}_frames"]).float()
        yk_key = f"Y{cs_ratio}_k_frames"
        self.Yk_frames = torch.from_numpy(data[yk_key]) if yk_key in data else None
        self.mask = torch.from_numpy(data[f"mask{cs_ratio}"])
        self.N = self.Y_frames.shape[-1]
        self.H_full = self.Y_frames.shape[1]
        self.n_frames = self.Y_frames.shape[0]
        self.op = MaskedRFFT2D(N=self.N, mask=self.mask.clone())

        self.group_id: Optional[torch.Tensor] = None
        if "group_id" in data:
            frame_gid = torch.from_numpy(data["group_id"]).long()
        else:
            frame_gid = torch.arange(self.n_frames, dtype=torch.long)

        if patch_h is not None and patch_h < self.H_full:
            stride = patch_stride or (patch_h // 2)
            self.patches: List[Tuple[int, int]] = []
            gids: List[int] = []
            for fi in range(self.n_frames):
                for rs in range(0, self.H_full - patch_h + 1, stride):
                    self.patches.append((fi, rs))
                    gids.append(int(frame_gid[fi]))
            self.patch_h = patch_h
            self.group_id = torch.tensor(gids, dtype=torch.long)
        else:
            self.patches = [(fi, 0) for fi in range(self.n_frames)]
            self.patch_h = self.H_full
            self.group_id = frame_gid

        self.device = device

    def to(self, device: str) -> "UltrasoundFrameDataset":
        self.device = device
        self.Y_frames = self.Y_frames.to(device)
        self.X_frames = self.X_frames.to(device)
        if self.Yk_frames is not None:
            self.Yk_frames = self.Yk_frames.to(device)
        self.mask = self.mask.to(device)
        self.op = MaskedRFFT2D(N=self.N, mask=self.mask.clone())
        if self.group_id is not None:
            self.group_id = self.group_id.to(device)
        return self

    def get_batch(self, idx) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """idx: 1D LongTensor of patch indices

        Returns:
            X_batch:  (B, 1, patch_h, W)  zero-filled
            Y_batch:  (B, 1, patch_h, W)  ground truth
            Yk_batch: (B, patch_h, K)     complex observations (or None)
        """
        patches_Y, patches_X, patches_Yk = [], [], []
        for i_raw in idx:
            i = i_raw.item() if isinstance(i_raw, torch.Tensor) else int(i_raw)
            fi, rs = self.patches[i]
            re = rs + self.patch_h
            patches_Y.append(self.Y_frames[fi, rs:re, :])
            patches_X.append(self.X_frames[fi, rs:re, :])
            if self.Yk_frames is not None:
                patches_Yk.append(self.Yk_frames[fi, rs:re, :])

        Y_batch = torch.stack(patches_Y).unsqueeze(1)
        X_batch = torch.stack(patches_X).unsqueeze(1)
        Yk_batch = torch.stack(patches_Yk) if patches_Yk else None
        return X_batch, Y_batch, Yk_batch

    def __len__(self) -> int:
        return len(self.patches)


class MultiDatasetSampler:
    """多数据集轮替采样器 (不同信号长度共同训练)

    每个 batch 从单个数据集中采样 (保证 batch 内信号长度一致),
    不同 batch 轮替使用不同数据集, 使所有数据均参与训练.
    """

    def __init__(self, datasets: List[UltrasoundFrameDataset],
                 val_ratio: float = 0.15, seed: int = 42,
                 split_mode: str = "group"):
        self.datasets = datasets
        self.train_indices: List[torch.Tensor] = []
        self.val_indices: List[torch.Tensor] = []

        global_group_offset = 0
        all_val_group_ids = []

        for ds in datasets:
            t_idx, v_idx = split_indices(
                len(ds), val_ratio, seed, split_mode, ds.group_id)
            self.train_indices.append(t_idx)
            self.val_indices.append(v_idx)

        self.n_train = sum(len(t) for t in self.train_indices)
        self.n_val = sum(len(v) for v in self.val_indices)
        self._rng = np.random.RandomState(seed)

    def to(self, device: str) -> "MultiDatasetSampler":
        for ds in self.datasets:
            ds.to(device)
        for i in range(len(self.train_indices)):
            self.train_indices[i] = self.train_indices[i].to(device)
            self.val_indices[i] = self.val_indices[i].to(device)
        return self

    def iter_train_batches(self, batch_size: int, epoch_seed: int = 0):
        """生成一个 epoch 的训练 batch, 轮替数据集

        Yields: (dataset, op, batch_indices)
        """
        rng = np.random.RandomState(epoch_seed)
        jobs = []
        for di, t_idx in enumerate(self.train_indices):
            perm = t_idx[torch.randperm(len(t_idx), device=t_idx.device)]
            for s in range(0, len(perm), batch_size):
                e = min(s + batch_size, len(perm))
                jobs.append((di, perm[s:e]))

        order = rng.permutation(len(jobs))
        for ji in order:
            di, idx = jobs[ji]
            ds = self.datasets[di]
            yield ds, ds.op, idx

    def iter_val_batches(self, batch_size: int):
        """生成验证 batch"""
        for di, v_idx in enumerate(self.val_indices):
            ds = self.datasets[di]
            for s in range(0, len(v_idx), batch_size):
                e = min(s + batch_size, len(v_idx))
                yield ds, ds.op, v_idx[s:e]

    @property
    def total_train_batches(self):
        return self.n_train

    def summary(self) -> str:
        lines = []
        for i, ds in enumerate(self.datasets):
            lines.append(
                f"  [{i}] N={ds.N}, 帧={ds.n_frames}, patch={len(ds)}, "
                f"train={len(self.train_indices[i])}, val={len(self.val_indices[i])}, "
                f"K={ds.op.K}"
            )
        return "\n".join(lines)


def split_indices(
    num_samples: int,
    val_ratio: float,
    seed: int = 42,
    split_mode: str = "random",
    group_id: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成训练/验证索引
    - random: 样本随机划分
    - group: 按 group_id 分组划分，避免同组泄漏
    """
    num_val = max(1, int(num_samples * val_ratio))
    rng = np.random.RandomState(seed)

    if split_mode == "group" and group_id is not None:
        gid = group_id.detach().cpu().numpy().astype(np.int64)
        unique_groups = np.unique(gid)
        rng.shuffle(unique_groups)

        val_mask = np.zeros(num_samples, dtype=bool)
        picked = 0
        for g in unique_groups:
            g_idx = np.where(gid == g)[0]
            val_mask[g_idx] = True
            picked += len(g_idx)
            if picked >= num_val:
                break

        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
    else:
        perm = rng.permutation(num_samples)
        val_idx = perm[:num_val]
        train_idx = perm[num_val:]

    train_idx = torch.from_numpy(np.sort(train_idx)).long()
    val_idx = torch.from_numpy(np.sort(val_idx)).long()
    return train_idx, val_idx
