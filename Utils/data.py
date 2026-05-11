import numpy as np
import torch
import torch.utils.data
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


@dataclass
class SpatialMask2D:
    """空间域通道掩膜算子: 丢弃部分阵元通道, 保留的通道信号不变.

    A:  (B, 1, H, W) → (B, K_ch, W)   选取 mask 标记的行
    At: (B, K_ch, W) → (B, 1, H, W)   将观测填回对应行, 其余补零
    """
    H: int
    mask: torch.Tensor

    def __post_init__(self):
        self.H = int(self.H)
        self.mask = self.mask.to(torch.bool)
        self.mu = torch.nonzero(self.mask, as_tuple=False).squeeze(1)

    @property
    def K(self) -> int:
        return int(self.mu.numel())

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) → (B, K_ch, W)"""
        return x.squeeze(1)[:, self.mu, :]

    def At(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, K_ch, W) → (B, 1, H, W)"""
        B, _, W = y.shape
        full = torch.zeros(B, self.H, W, device=y.device, dtype=y.dtype)
        full[:, self.mu, :] = y
        return full.unsqueeze(1)


@dataclass
class SpatialMask2D:
    """空间域通道掩膜算子: 模拟稀疏阵列 (丢弃部分阵元通道)

    每帧 RF 数据为 (H, W), H 为阵元数, W 为时间采样数.
    mask 标记哪些阵元 (行) 被保留, 其余行被丢弃.

    接口与 MaskedRFFT2D 对称:
        A:  (B, 1, H, W) → (B, K_ch, W)   选取保留通道
        At: (B, K_ch, W) → (B, 1, H, W)   零填充缺失通道
    """
    H: int
    mask: torch.Tensor

    def __post_init__(self):
        self.H = int(self.H)
        self.mask = self.mask.to(torch.bool)
        self.mu = torch.nonzero(self.mask, as_tuple=False).squeeze(1)

    @property
    def K(self) -> int:
        return int(self.mu.numel())

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) → (B, K_ch, W)"""
        return x.squeeze(1).index_select(dim=1, index=self.mu)

    def At(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, K_ch, W) → (B, 1, H, W)"""
        B, _, W = y.shape
        full = torch.zeros(B, self.H, W, device=y.device, dtype=y.dtype)
        full[:, self.mu, :] = y
        return full.unsqueeze(1)


class UltrasoundDataset(torch.utils.data.Dataset):
    """从预处理 npz 文件加载超声数据集.

    数据保留在 CPU (pin_memory), 仅 mask/op 搬到目标 device.
    配合 DataLoader(pin_memory=True) + non_blocking 实现 CPU-GPU 异步传输.
    """

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
        self.c = float(data["c"]) if "c" in data else 1540.0
        self.op = MaskedRFFT1D(N=self.N, mask=self.mask.clone())
        self.device = device
        self.has_yk = self.Y_k is not None

        self.rf_data_3d = data["rf_data_3d"] if "rf_data_3d" in data else None
        self.angles = data["angles"] if "angles" in data else None
        self.probe_geometry = data["probe_geometry"] if "probe_geometry" in data else None
        self.initial_time = float(data["initial_time"]) if "initial_time" in data else None
        self.scan_x_axis = data["scan_x_axis"] if "scan_x_axis" in data else None
        self.scan_z_axis = data["scan_z_axis"] if "scan_z_axis" in data else None
        self.selected_angles = data["selected_angles"] if "selected_angles" in data else None

        if torch.cuda.is_available():
            self.Y = self.Y.pin_memory()
            self.X = self.X.pin_memory()
            if self.Y_k is not None:
                self.Y_k = self.Y_k.pin_memory()

    def to(self, device: str) -> "UltrasoundDataset":
        """只搬 mask/op 到 device, 数据留 CPU 以节省显存."""
        self.device = device
        self.mask = self.mask.to(device)
        self.op = MaskedRFFT1D(N=self.N, mask=self.mask.clone())
        return self

    def __getitem__(self, idx):
        """单样本访问, 供 DataLoader 使用."""
        if self.has_yk:
            return self.X[idx], self.Y[idx], self.Y_k[idx]
        return self.X[idx], self.Y[idx]

    def get_batch(self, idx, device=None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x, y = self.X[idx], self.Y[idx]
        yk = self.Y_k[idx] if self.has_yk else None
        if device is not None:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if yk is not None:
                yk = yk.to(device, non_blocking=True)
        return x, y, yk

    def __len__(self) -> int:
        return self.Y.shape[0]


class UltrasoundFrameDataset:
    """帧级超声数据集 (2D), 支持 patch 提取

    从 prepare_picmus_data.py --mode frame 生成的 npz 加载.
    每个样本是一个 (patch_h, W) 的 2D RF 帧片段.

    DAS 元数据 (angles, probe_geometry, scan grid 等) 会在 npz 中
    可选地加载, 供训练时可微 DAS 图像域损失使用.
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

        self._compress_mode = str(data["compress_mode"]) if "compress_mode" in data else "freq"
        if self._compress_mode == "spatial":
            self.Yk_frames = None
            self.op = SpatialMask2D(H=self.H_full, mask=self.mask.clone())
        else:
            # "freq" and "post_das" both use MaskedRFFT2D
            self.op = MaskedRFFT2D(N=self.N, mask=self.mask.clone())

        self.group_id: Optional[torch.Tensor] = None
        if "group_id" in data:
            frame_gid = torch.from_numpy(data["group_id"]).long()
        else:
            frame_gid = torch.arange(self.n_frames, dtype=torch.long)

        if patch_h is not None and patch_h < self.H_full:
            if self._compress_mode == "spatial":
                raise ValueError(
                    f"Spatial masking 不支持 patch (patch_h={patch_h} < H={self.H_full}). "
                    f"请设置 --patch_h {self.H_full} 或不指定, 使用全帧训练.")
            stride = patch_stride or (patch_h // 2)
            frame_indices = []
            row_starts = []
            gids: List[int] = []
            for fi in range(self.n_frames):
                for rs in range(0, self.H_full - patch_h + 1, stride):
                    frame_indices.append(fi)
                    row_starts.append(rs)
                    gids.append(int(frame_gid[fi]))
            self.patch_h = patch_h
            self._patch_frame_idx = torch.tensor(frame_indices, dtype=torch.long)
            self._patch_row_start = torch.tensor(row_starts, dtype=torch.long)
            self.group_id = torch.tensor(gids, dtype=torch.long)
        else:
            self._patch_frame_idx = torch.arange(self.n_frames, dtype=torch.long)
            self._patch_row_start = torch.zeros(self.n_frames, dtype=torch.long)
            self.patch_h = self.H_full
            self.group_id = frame_gid

        self._n_patches = len(self._patch_frame_idx)
        self.device = device

        if torch.cuda.is_available():
            self.Y_frames = self.Y_frames.pin_memory()
            self.X_frames = self.X_frames.pin_memory()
            if self.Yk_frames is not None:
                self.Yk_frames = self.Yk_frames.pin_memory()

        # --- DAS 元数据 (可选) ---
        self._load_das_meta(data)

    def _load_das_meta(self, data):
        """从 npz 加载 DAS 所需的几何和角度信息."""
        self.angles: Optional[np.ndarray] = None
        self.frame_angle_vals: Optional[np.ndarray] = None
        self.probe_geometry: Optional[np.ndarray] = None
        self.initial_time: Optional[float] = None
        self.scan_x_axis: Optional[np.ndarray] = None
        self.scan_z_axis: Optional[np.ndarray] = None
        self.fs: Optional[float] = None
        self.c: Optional[float] = None

        if "angles" not in data:
            return

        angles = np.asarray(data["angles"], dtype=np.float64)
        self.angles = angles
        self.fs = float(data["fs"]) if "fs" in data else None
        self.c = float(data["c"]) if "c" in data else 1540.0

        if "probe_geometry" in data:
            self.probe_geometry = np.asarray(data["probe_geometry"])
        if "initial_time" in data:
            self.initial_time = float(data["initial_time"])
        if "scan_x_axis" in data:
            self.scan_x_axis = np.asarray(data["scan_x_axis"])
        if "scan_z_axis" in data:
            self.scan_z_axis = np.asarray(data["scan_z_axis"])

        # 建立 frame -> angle 映射 (必须保证 len == n_frames)
        if "frame_angle_idx" in data:
            fai = np.asarray(data["frame_angle_idx"], dtype=int)
            if len(fai) == self.n_frames:
                self.frame_angle_vals = angles[fai]
        elif "selected_angles" in data:
            sel = np.asarray(data["selected_angles"], dtype=int)
            if len(sel) == self.n_frames:
                self.frame_angle_vals = angles[sel]
        else:
            if len(angles) == self.n_frames:
                self.frame_angle_vals = angles

    @property
    def has_das_meta(self) -> bool:
        """DAS 图像域损失是否可用."""
        return (self.frame_angle_vals is not None
                and self.probe_geometry is not None
                and self.initial_time is not None
                and self.scan_x_axis is not None
                and self.scan_z_axis is not None)

    def to(self, device: str) -> "UltrasoundFrameDataset":
        """只搬 mask/op 到 device, 帧数据留 CPU 以节省显存."""
        self.device = device
        self.mask = self.mask.to(device)
        if self._compress_mode == "spatial":
            self.op = SpatialMask2D(H=self.H_full, mask=self.mask.clone())
        else:
            # "freq" and "post_das" both use MaskedRFFT2D
            self.op = MaskedRFFT2D(N=self.N, mask=self.mask.clone())
        return self

    def get_batch(self, idx, device=None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """idx: 1D LongTensor of patch indices

        Returns:
            X_batch:  (B, 1, patch_h, W)  zero-filled
            Y_batch:  (B, 1, patch_h, W)  ground truth
            Yk_batch: (B, patch_h, K)     complex observations (or None)
        """
        if isinstance(idx, torch.Tensor):
            idx_cpu = idx.cpu()
        else:
            idx_cpu = torch.as_tensor(idx, dtype=torch.long)

        fi = self._patch_frame_idx[idx_cpu]
        rs = self._patch_row_start[idx_cpu]

        if self.patch_h == self.H_full:
            Y_batch = self.Y_frames[fi].unsqueeze(1)
            X_batch = self.X_frames[fi].unsqueeze(1)
            Yk_batch = self.Yk_frames[fi] if self.Yk_frames is not None else None
        else:
            row_offsets = torch.arange(self.patch_h, dtype=torch.long)
            rows = rs.unsqueeze(1) + row_offsets.unsqueeze(0)  # (B, patch_h)
            Y_batch = self.Y_frames[fi.unsqueeze(1).expand_as(rows), rows].unsqueeze(1)
            X_batch = self.X_frames[fi.unsqueeze(1).expand_as(rows), rows].unsqueeze(1)
            if self.Yk_frames is not None:
                Yk_batch = self.Yk_frames[fi.unsqueeze(1).expand_as(rows), rows]
            else:
                Yk_batch = None

        if device is not None:
            Y_batch = Y_batch.to(device, non_blocking=True)
            X_batch = X_batch.to(device, non_blocking=True)
            if Yk_batch is not None:
                Yk_batch = Yk_batch.to(device, non_blocking=True)
        return X_batch, Y_batch, Yk_batch

    def get_frame_angles(self, idx, device=None) -> Optional[torch.Tensor]:
        """返回 batch 中每个 patch 对应的平面波角度 (rad).

        idx: 1D LongTensor of patch indices
        Returns: (B,) float64 tensor, or None if DAS metadata unavailable
        """
        if self.frame_angle_vals is None:
            return None
        if isinstance(idx, torch.Tensor):
            idx_cpu = idx.cpu()
        else:
            idx_cpu = torch.as_tensor(idx, dtype=torch.long)
        fi = self._patch_frame_idx[idx_cpu]
        out = torch.from_numpy(self.frame_angle_vals[fi.numpy()]).to(torch.float64)
        if device is not None:
            out = out.to(device, non_blocking=True)
        return out

    def __len__(self) -> int:
        return self._n_patches


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
        """只搬 mask/op 到 device, 帧数据和索引留 CPU."""
        self.device = device
        for ds in self.datasets:
            ds.to(device)
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


class CUDAPrefetcher:
    """使用独立 CUDA stream 预取下一个 batch, 实现 CPU-GPU 数据传输与计算的重叠."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        it = iter(self.loader)
        batch = self._to_device(self._next_or_none(it))
        while batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            current = batch
            batch = self._to_device(self._next_or_none(it))
            yield current

    @staticmethod
    def _next_or_none(it):
        try:
            return next(it)
        except StopIteration:
            return None

    def _to_device(self, batch):
        if batch is None:
            return None
        with torch.cuda.stream(self.stream):
            return tuple(
                t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in batch
            )


def _collate_with_optional_yk(batch):
    """自定义 collate: 处理 2 元素 (无 Y_k) 和 3 元素 (有 Y_k) 的情况."""
    if len(batch[0]) == 3:
        xs, ys, yks = zip(*batch)
        return torch.stack(xs), torch.stack(ys), torch.stack(yks)
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys), None


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
