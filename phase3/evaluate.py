# Evaluate a trained SurfaceGraph-ShapeEmbed model.
#
#   python phase3/evaluate.py --encoder set --features both          # one run: metrics + figures
#   python phase3/evaluate.py --encoder set --features both --ae
#   python phase3/evaluate.py --compare                              # table over all trained checkpoints
#
# Per-run mode produces, under outputs/shapes3d/:
#   results/<run>_evaluation.json   -- 5-fold LR accuracy on the latent (paper-style),
#                                      permutation-robustness (latent drift + permuted-input accuracy),
#                                      mean reconstruction error, mean EDM-validity residual
#   figures/<run>_reconstruction.png  -- decoded point cloud rendered as its convex-hull surface, vs ground truth
#   figures/<run>_interpolation.png   -- latent interpolation between two shapes, decoded as surfaces
#
# --compare mode writes results/encoder_comparison.{json,csv}.

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from common import (CKPT_DIR, CLASSES, DEVICE, FIG_DIR, RES_DIR, Shapes3DTensors, resolve_device,
                    classical_mds, convex_hull_faces, set_seed)
from models import build_model, edm_rank3_penalty, sorted_recon_loss
from train import FEATURES, make_inputs


# ---------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------

def load_run(name: str):
    ckpt = torch.load(CKPT_DIR / f"{name}_best.pt", map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = build_model(a["encoder"], ckpt["channels"], ckpt["n_points"], a["latent_dim"],
                        variational=not a["ae"], decoder=a.get("decoder", "point"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)
    return model, a, ckpt


@torch.no_grad()
def encode(model, x):
    out = []
    for i in range(0, len(x), 256):
        out.append(model(x[i:i + 256].to(DEVICE))["mu"].cpu())
    return torch.cat(out).numpy()


def permute_inputs(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Apply an independent random point permutation to each sample (rows & columns)."""
    g = torch.Generator().manual_seed(seed)
    out = torch.empty_like(x)
    n = x.shape[-1]
    for i in range(len(x)):
        p = torch.randperm(n, generator=g)
        out[i] = x[i][:, p][:, :, p]
    return out


# ---------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------

def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def cv_accuracy(Z: np.ndarray, y: np.ndarray, seed: int = 42):
    n_splits = min(5, int(np.bincount(y).min()))
    skf = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)
    accs = [_lr().fit(Z[tr], y[tr]).score(Z[te], y[te]) for tr, te in skf.split(Z, y)]
    return float(np.mean(accs)), float(np.std(accs))


def permuted_input_accuracy(Z_tr, y_tr, Z_te_perm, y_te):
    return float(_lr().fit(Z_tr, y_tr).score(Z_te_perm, y_te))


# ---------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------

def _canonical_orient(P):
    """Centre P and rotate it into its principal-axis frame, with a deterministic
    sign convention -- so two point clouds of the same shape (possibly in different
    poses, possibly in different point orderings) are drawn the same way."""
    P = P - P.mean(0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    P = P @ Vt.T
    for k in range(3):                                  # fix the per-axis sign by third moment
        if np.sum(P[:, k] ** 3) < 0:
            P[:, k] = -P[:, k]
    return P


def _render_shape(ax, P, title, color=(0.30, 0.45, 0.69), points=True):
    """Draw a point cloud as a *surface*: the faces of its convex hull (all shape
    families here are convex), Lambert-shaded so flat faces read as flat, with the
    sample points on top.  The cloud is first put in a canonical orientation."""
    import matplotlib.colors as mcolors
    P = _canonical_orient(P)
    faces = convex_hull_faces(P)
    if faces is not None:
        tris = P[faces]                                              # (F, 3, 3)
        nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
        light = np.array([0.4, 0.3, 1.0]); light /= np.linalg.norm(light)
        shade = 0.55 + 0.45 * np.clip(np.abs(nrm @ light), 0, 1)     # [0.55, 1.0]
        base = np.array(mcolors.to_rgb(color))
        cols = np.clip(base[None, :] * shade[:, None], 0, 1)
        pc = Poly3DCollection(tris, facecolors=cols, edgecolor=(1, 1, 1, 0.5), linewidths=0.3)
        ax.add_collection3d(pc)
    if points:
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=5, c="#1a1a1a", depthshade=False)
    ax.set_title(title, fontsize=9, pad=-2)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=40)
    r = float(np.abs(P).max()) * 0.92 + 1e-9
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _recon_points(model, x_batch):
    """Reconstructed point cloud per sample: the decoder's P_hat if it has one,
    otherwise classical MDS of the decoded distance matrix."""
    out = model(x_batch.to(DEVICE))
    if out["P_hat"] is not None:
        return out["P_hat"].cpu().numpy()
    D_hat = out["D_hat"].cpu().numpy()
    return np.stack([classical_mds(D) for D in D_hat])


@torch.no_grad()
def fig_reconstruction(model, x, P_true, y, path, seed=0):
    idx = [int(np.where(y == c)[0][0]) for c in range(len(CLASSES))]
    P_hat = _recon_points(model, x[idx])
    fig = plt.figure(figsize=(3.4 * len(idx), 6.8))
    for j, i in enumerate(idx):
        ax = fig.add_subplot(2, len(idx), j + 1, projection="3d")
        _render_shape(ax, P_true[i], f"{CLASSES[y[i]]}\n(ground truth)", color="#55A868")
        ax = fig.add_subplot(2, len(idx), len(idx) + j + 1, projection="3d")
        _render_shape(ax, P_hat[j], "reconstruction", color="#4C72B0")
    fig.suptitle("Reconstruction:  decoder output (point cloud) rendered as its convex-hull surface "
                 "(both in canonical principal-axis orientation)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


@torch.no_grad()
def fig_interpolation(model, x, y, path, classes=(0, 1), steps=7):
    ca, cb = classes
    ia = int(np.where(y == ca)[0][0]); ib = int(np.where(y == cb)[0][0])
    mu = model(x[[ia, ib]].to(DEVICE))["mu"].cpu().numpy()
    fig = plt.figure(figsize=(2.9 * steps, 3.2))
    for k in range(steps):
        t = k / (steps - 1)
        z = torch.from_numpy(((1 - t) * mu[0] + t * mu[1])[None]).float().to(DEVICE)
        dec = model.decoder(z)
        P = dec["P_hat"].cpu().numpy()[0] if dec["P_hat"] is not None else classical_mds(dec["D_hat"].cpu().numpy()[0])
        ax = fig.add_subplot(1, steps, k + 1, projection="3d")
        _render_shape(ax, P, f"t = {t:.2f}", color="#4C72B0")        # _render_shape canonicalises orientation
    fig.suptitle(f"Latent interpolation:  {CLASSES[ca]}  ->  {CLASSES[cb]}   "
                 f"(decoded shapes, convex-hull surfaces)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ---------------------------------------------------------------------
# per-run evaluation
# ---------------------------------------------------------------------

def evaluate_run(name: str, seed: int = 42):
    set_seed(seed)
    model, a, ckpt = load_run(name)
    tr, te = Shapes3DTensors("train"), Shapes3DTensors("test")
    x_tr, _, _ = make_inputs(tr, a["features"])
    x_te, d_te, p_te = make_inputs(te, a["features"])

    Z_tr, Z_te = encode(model, x_tr), encode(model, x_te)
    acc_mean, acc_std = cv_accuracy(Z_te, te.y, seed)

    x_te_perm = permute_inputs(x_te, seed=seed)
    Z_te_perm = encode(model, x_te_perm)
    drift = float(np.mean(np.linalg.norm(Z_te_perm - Z_te, axis=1) / (np.linalg.norm(Z_te, axis=1) + 1e-9)))
    acc_perm = permuted_input_accuracy(Z_tr, tr.y, Z_te_perm, te.y)

    with torch.no_grad():
        D_hat_te = torch.cat([model(x_te[i:i + 256].to(DEVICE))["D_hat"].cpu() for i in range(0, len(x_te), 256)])
        edm = edm_rank3_penalty(D_hat_te).item()
        rec = sorted_recon_loss(D_hat_te, d_te).item()             # sorted pairwise-distance MSE

    res = {"run": name, "args": a, "n_params": sum(p.numel() for p in model.parameters()),
           "cv_accuracy_mean": acc_mean, "cv_accuracy_std": acc_std,
           "latent_drift_under_permutation": drift, "accuracy_with_permuted_inputs": acc_perm,
           "test_recon_error": rec, "test_edm_residual": edm}
    with open(RES_DIR / f"{name}_evaluation.json", "w") as f:
        json.dump(res, f, indent=2)

    fig_reconstruction(model, x_te, te.P, te.y, FIG_DIR / f"{name}_reconstruction.png")
    fig_interpolation(model, x_te, te.y, FIG_DIR / f"{name}_interpolation.png")

    print(f"[{name}]  5-fold LR acc {acc_mean:.3f}±{acc_std:.3f}   "
          f"latent drift under point-permutation {drift:.4f}   acc(permuted inputs) {acc_perm:.3f}\n"
          f"          test recon err {rec:.3e}   EDM residual {edm:.3e}   figures -> {FIG_DIR}")
    return res


# ---------------------------------------------------------------------
# comparison table over all trained checkpoints
# ---------------------------------------------------------------------

def compare(seed: int = 42):
    runs = sorted(p.stem[:-5] for p in CKPT_DIR.glob("*_best.pt"))
    if not runs:
        print("no checkpoints found in", CKPT_DIR); return
    rows = [evaluate_run(r, seed) for r in runs]
    keys = ["run", "cv_accuracy_mean", "cv_accuracy_std", "latent_drift_under_permutation",
            "accuracy_with_permuted_inputs", "test_recon_error", "test_edm_residual"]
    with open(RES_DIR / "encoder_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(RES_DIR / "encoder_comparison.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print("\n=== comparison ===")
    hdr = f"{'run':<22s} {'acc':>12s} {'drift(perm)':>12s} {'acc(perm in)':>13s} {'recon err':>12s} {'EDM resid':>12s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:<22s} {r['cv_accuracy_mean']:.3f}±{r['cv_accuracy_std']:.2f}  "
              f"{r['latent_drift_under_permutation']:>12.4f} {r['accuracy_with_permuted_inputs']:>13.3f} "
              f"{r['test_recon_error']:>12.2e} {r['test_edm_residual']:>12.2e}")
    print(f"\nwrote {RES_DIR/'encoder_comparison.csv'}")


def main():
    global DEVICE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=["set", "cnn", "mlp"], default="set")
    ap.add_argument("--features", choices=list(FEATURES), default="both")
    ap.add_argument("--ae", action="store_true")
    ap.add_argument("--run", type=str, default=None, help="evaluate this exact run name (overrides --encoder/--features/--ae)")
    ap.add_argument("--compare", action="store_true", help="evaluate every trained checkpoint and tabulate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = ap.parse_args()
    try:
        DEVICE = resolve_device(args.device)
    except ValueError as exc:
        ap.error(str(exc))
    if args.compare:
        compare(args.seed)
    else:
        name = args.run or (f"{args.encoder}_{args.features}" + ("_ae" if args.ae else ""))
        evaluate_run(name, args.seed)


if __name__ == "__main__":
    main()
