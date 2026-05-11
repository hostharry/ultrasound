import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

# ====== 1) 改成你的三张图路径（顺序：Y, X8, X15 或你想比较的顺序）======
paths = [
    "Figure_1.png",  # 例如：GT (Y)
    "Figure_2.png",  # 例如：X8
    "Figure_3.png",  # 例如：X15
]
names = ["GT(Y)", "X8", "X15"]

# ====== 工具函数 ======
def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)

def to_gray(rgb):
    # 标准亮度加权，避免只取一个通道导致偏差
    return (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.float32)

def mse(a, b):
    return float(np.mean((a - b) ** 2))

def psnr(a, b, data_range=255.0):
    m = mse(a, b)
    if m < 1e-12:
        return 99.0
    return float(20*np.log10(data_range) - 10*np.log10(m))

# SSIM：优先用 skimage（推荐）；没有就只输出 PSNR/MSE
def try_ssim(a, b, data_range=255.0):
    try:
        from skimage.metrics import structural_similarity as ssim
        val = ssim(a, b, data_range=data_range)
        return float(val)
    except Exception:
        return None

# ====== 2) 载入图像 ======
imgs_rgb = [load_rgb(p) for p in paths]
assert all(im.shape == imgs_rgb[0].shape for im in imgs_rgb), "三张图尺寸必须一致（你的看起来是一致的）"

# ====== 3) 在第一张图上手动框选 ROI ======
roi = {"x1": None, "y1": None, "x2": None, "y2": None}

fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(imgs_rgb[0])
ax.set_title("Drag to select ROI (ONLY the B-mode image area, exclude axes/title/colorbar). Close window when done.")

def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    roi["x1"], roi["y1"] = min(x1, x2), min(y1, y2)
    roi["x2"], roi["y2"] = max(x1, x2), max(y1, y2)

rect = RectangleSelector(
    ax, onselect,
    useblit=True,
    button=[1],  # 左键
    minspanx=10, minspany=10,
    spancoords="pixels",
    interactive=True
)

plt.show()

x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
if None in [x1, y1, x2, y2]:
    raise RuntimeError("没有选中 ROI。请重新运行并用鼠标拖拽框选区域。")

print(f"[ROI] x: {x1}~{x2}, y: {y1}~{y2}")

# ====== 4) 统一裁剪 + 转灰度 ======
crops_gray = []
for im in imgs_rgb:
    crop = im[y1:y2, x1:x2, :]
    crops_gray.append(to_gray(crop))

# ====== 5) 指标：以 GT(Y) 为参考，比较 X8/X15 ======
ref = crops_gray[0]
data_range = 255.0

print("\n=== Metrics vs GT(Y) on cropped ROI (PNG-domain) ===")
for i in range(1, len(crops_gray)):
    cur = crops_gray[i]
    m = mse(ref, cur)
    p = psnr(ref, cur, data_range=data_range)
    s = try_ssim(ref, cur, data_range=data_range)
    if s is None:
        print(f"{names[i]}: MSE={m:.3e}, PSNR={p:.2f} dB, SSIM= (skimage not found)")
    else:
        print(f"{names[i]}: MSE={m:.3e}, PSNR={p:.2f} dB, SSIM={s:.4f}")

# ====== 6) 差分可视化（|X - Y|） ======
fig, axes = plt.subplots(1, len(crops_gray), figsize=(15, 4))
for ax, g, nm in zip(axes, crops_gray, names):
    ax.imshow(g, cmap="gray", vmin=0, vmax=255)
    ax.set_title(nm)
    ax.axis("off")
plt.suptitle("Cropped ROI (grayscale)")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, len(crops_gray)-1, figsize=(10, 4))
for j, i in enumerate(range(1, len(crops_gray))):
    diff = np.abs(crops_gray[i] - ref)
    axes[j].imshow(diff, cmap="hot")
    axes[j].set_title(f"|{names[i]} - GT|")
    axes[j].axis("off")
plt.suptitle("Absolute difference heatmaps (PNG-domain)")
plt.tight_layout()
plt.show()

# ====== 7) 快速剖面对比（可选）：固定某一条 depth 行 / line 列 ======
# 这里用“ROI 中间的深度位置”做一条横向剖面
mid_y = ref.shape[0] // 2
plt.figure(figsize=(8,4))
plt.plot(ref[mid_y], label="GT(Y)")
for i in range(1, len(crops_gray)):
    plt.plot(crops_gray[i][mid_y], label=names[i], alpha=0.8)
plt.title("Profile at middle depth (PNG-domain)")
plt.legend()
plt.tight_layout()
plt.show()
