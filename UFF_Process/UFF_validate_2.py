#!/usr/bin/env python3
"""
Evaluate band-limited (downsampled-in-frequency) ultrasound dataset saved as NPZ.

Usage:
  python evaluate_downsample.py dataset_fdbf_energy_mu_8_9_15.npz

It will:
  - load X8/X9/X15/Y, fs/fc/c, mu*
  - make B-mode images (Hilbert envelope + log compression)
  - report PSNR/SSIM/MSE on display-domain images (0..1)
  - save comparison figures to ./outputs/

Optional (manual ROI CNR):
  - run with --roi to interactively draw two rectangles on the GT image
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def bmode_db(lines, ref_max=None, db_min=-60.0, eps=1e-12):
    analytic = hilbert(lines, axis=1)
    env = np.abs(analytic)
    if ref_max is None:
        ref_max = float(np.max(env)) + eps
    envn = env / ref_max
    db = 20.0 * np.log10(envn + eps)
    db = np.clip(db, db_min, 0.0)
    img01 = (db - db_min) / (0.0 - db_min)
    return img01.astype(np.float32), db.astype(np.float32), env.astype(np.float32), ref_max

def metrics(img, ref):
    mse = float(np.mean((img - ref) ** 2))
    p = float(psnr(ref, img, data_range=1.0))
    s = float(ssim(ref, img, data_range=1.0))
    return mse, p, s

def cnr(roi_a, roi_b, eps=1e-12):
    ma, sa = float(np.mean(roi_a)), float(np.std(roi_a))
    mb, sb = float(np.mean(roi_b)), float(np.std(roi_b))
    return abs(ma - mb) / (np.sqrt(sa*sa + sb*sb) + eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=str)
    ap.add_argument("--roi", action="store_true", help="interactive CNR ROI selection on GT")
    ap.add_argument("--dbmin", type=float, default=-60.0)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    Y = d["Y"].astype(np.float32)
    X8 = d["X8"].astype(np.float32)
    X9 = d["X9"].astype(np.float32) if "X9" in d else None
    X15 = d["X15"].astype(np.float32)
    fs = float(d["fs"]); fc = float(d["fc"]); c = float(d["c"])
    L, N = Y.shape

    t = np.arange(N) / fs
    depth_mm = (c * t / 2.0) * 1e3

    Y01, Ydb, Yenv, ref = bmode_db(Y, db_min=args.dbmin)
    X801, X8db, X8env, _ = bmode_db(X8, ref_max=ref, db_min=args.dbmin)
    X1501, X15db, X15env, _ = bmode_db(X15, ref_max=ref, db_min=args.dbmin)
    if X9 is not None:
        X901, X9db, X9env, _ = bmode_db(X9, ref_max=ref, db_min=args.dbmin)

    # central ROI
    x0, x1 = int(0.1*L), int(0.9*L)
    z0 = int(np.searchsorted(depth_mm, 5.0))
    z1 = int(np.searchsorted(depth_mm, 55.0))
    if z1 <= z0 + 10:
        z0, z1 = int(0.05*N), int(0.95*N)
    roi = (slice(x0, x1), slice(z0, z1))

    print(f"[info] L={L}, N={N}, fs={fs/1e6:.2f} MHz, fc={fc/1e6:.2f} MHz, c={c:.1f} m/s")
    for name, img in [("X8", X801), ("X15", X1501)] + ([("X9", X901)] if X9 is not None else []):
        mse, p, s = metrics(img, Y01)
        mse_r, p_r, s_r = metrics(img[roi], Y01[roi])
        print(f"[{name}] full: MSE={mse:.6f}, PSNR={p:.2f} dB, SSIM={s:.4f} | ROI: MSE={mse_r:.6f}, PSNR={p_r:.2f} dB, SSIM={s_r:.4f}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # B-mode comparison
    imgs = [("GT (Y)", Ydb), ("Input X8", X8db)]
    if X9 is not None: imgs.append(("Input X9", X9db))
    imgs.append(("Input X15", X15db))

    ncols = 2
    nrows = int(np.ceil(len(imgs)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4*nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    extent = [0, L-1, float(depth_mm[-1]), float(depth_mm[0])]
    for ax, (title, dbimg) in zip(axes, imgs):
        im = ax.imshow(dbimg.T, cmap="gray", vmin=args.dbmin, vmax=0, aspect="auto", origin="upper", extent=extent)
        ax.set_title(title)
        ax.set_xlabel("line index"); ax.set_ylabel("depth (mm)")
        rect = Rectangle((x0, depth_mm[z0]), x1-x0, depth_mm[z1]-depth_mm[z0], fill=False, linewidth=1.2)
        ax.add_patch(rect)
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.9, label="dB (normalized to max(Y))")
    fig.savefig(out_dir / "bmode_compare.png", dpi=160)

    # spectrum
    f = np.fft.rfftfreq(N, d=1.0/fs)
    def avg_rfft_mag(lines):
        return np.mean(np.abs(np.fft.rfft(lines, axis=1)), axis=0)

    def dbnorm(x):
        return 20*np.log10(x/np.max(x) + 1e-12)

    magY = avg_rfft_mag(Y)
    mag8 = avg_rfft_mag(X8)
    mag15 = avg_rfft_mag(X15)
    plt.figure(figsize=(10,4))
    plt.plot(f/1e6, dbnorm(magY), label="Y")
    plt.plot(f/1e6, dbnorm(mag8), label="X8")
    if X9 is not None: plt.plot(f/1e6, dbnorm(avg_rfft_mag(X9)), label="X9")
    plt.plot(f/1e6, dbnorm(mag15), label="X15")
    plt.xlabel("frequency (MHz)"); plt.ylabel("magnitude (dB, normalized)")
    plt.ylim(-80, 5); plt.xlim(0, fs/2/1e6)
    plt.title("Average rFFT magnitude (per-line average)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "avg_spectrum.png", dpi=160)

    print(f"[saved] {out_dir}/bmode_compare.png, {out_dir}/avg_spectrum.png")

    if args.roi:
        # Interactive CNR on GT envelope (linear) in ROI display.
        # NOTE: For CNR you usually want two ROIs: lesion region and background region.
        fig, ax = plt.subplots(figsize=(7,5))
        ax.imshow(Ydb.T, cmap="gray", vmin=args.dbmin, vmax=0, aspect="auto", origin="upper", extent=extent)
        ax.set_title("Draw 2 rectangles: (1) target/lesion, (2) background. Close window to finish.")
        ax.set_xlabel("line index"); ax.set_ylabel("depth (mm)")

        rois = []
        print("[roi] Click 2 corners for ROI#1, then 2 corners for ROI#2 ...")
        for k in range(2):
            pts = plt.ginput(2, timeout=-1)
            (xA,yA),(xB,yB) = pts
            x_min, x_max = sorted([xA,xB]); y_min, y_max = sorted([yA,yB])
            rois.append((x_min,x_max,y_min,y_max))
            ax.add_patch(Rectangle((x_min,y_min), x_max-x_min, y_max-y_min, fill=False, linewidth=2))
            fig.canvas.draw()

        plt.show()

        # Convert y(mm) back to sample index
        def mm_to_z(mm): return int(np.clip(np.searchsorted(depth_mm, mm), 0, N-1))
        def x_to_i(x): return int(np.clip(round(x), 0, L-1))

        (x_min,x_max,y_min,y_max) = rois[0]
        i0,i1 = sorted([x_to_i(x_min), x_to_i(x_max)])
        z0,z1 = sorted([mm_to_z(y_min), mm_to_z(y_max)])
        A = Yenv[i0:i1+1, z0:z1+1]

        (x_min,x_max,y_min,y_max) = rois[1]
        i0,i1 = sorted([x_to_i(x_min), x_to_i(x_max)])
        z0,z1 = sorted([mm_to_z(y_min), mm_to_z(y_max)])
        B = Yenv[i0:i1+1, z0:z1+1]

        print(f"[CNR] (linear envelope): {cnr(A,B):.3f}")

if __name__ == "__main__":
    main()
