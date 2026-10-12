# Train one SurfaceGraph-ShapeEmbed (V)AE on the synthetic 3D dataset.
#
#   python phase3/train.py                              # proposed model: set encoder, Euclidean+geodesic, VAE
#   python phase3/train.py --encoder set --features euclidean
#   python phase3/train.py --encoder cnn --features both        # ablation baseline
#   python phase3/train.py --ae                                 # deterministic AE variant
#
# Writes  outputs/shapes3d/checkpoints/<run>_best.pt
#   and   outputs/shapes3d/results/<run>_history.json
# where <run> = "<encoder>_<features>[_ae]".

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import CKPT_DIR, DEVICE, RES_DIR, Shapes3DTensors, set_seed, resolve_device
from models import aligned_point_loss, build_model, edm_rank3_penalty, kl_loss, sorted_recon_loss

FEATURES = {"euclidean": ("D_e",), "geo": ("D_g",), "both": ("D_e", "D_g")}


def make_inputs(t: Shapes3DTensors, features: str):
    """Return (x, D_target, P_target): x is (M, C, N, N), D_target the Euclidean EDM
    (M, N, N), P_target the target point cloud (M, N, 3)."""
    chans = [getattr(t, c) for c in FEATURES[features]]
    x = np.stack(chans, axis=1).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(t.D_e), torch.from_numpy(t.P)


def _recon_metric(out, D_tgt, P_tgt):
    """Reconstruction term: sorted pairwise-distance MSE (permutation- and
    rotation-invariant), plus a small alignment-invariant
    point-matching term that sharpens the point decoder's output once it is roughly
    the right shape."""
    rec = sorted_recon_loss(out["D_hat"], D_tgt)
    if out["P_hat"] is not None:
        rec = rec + 0.05 * aligned_point_loss(out["P_hat"], P_tgt)
    return rec


def run_name(args) -> str:
    return (f"{args.encoder}_{args.features}"
            + ("_mat" if args.decoder == "matrix" else "")
            + ("_ae" if args.ae else "")
            + (f"_{args.tag}" if args.tag else ""))


def evaluate_recon(model, loader):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, dtgt, ptgt in loader:
            x, dtgt, ptgt = x.to(DEVICE), dtgt.to(DEVICE), ptgt.to(DEVICE)
            tot += _recon_metric(model(x), dtgt, ptgt).item() * len(x)
            n += len(x)
    return tot / n


def main():
    global DEVICE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=["set", "cnn", "mlp"], default="set")
    ap.add_argument("--features", choices=list(FEATURES), default="both")
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=350)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--decoder", choices=["point", "matrix"], default="point",
                    help="point: decode an N×3 cloud (EDM valid by construction); matrix: decode N×N directly")
    ap.add_argument("--ae", action="store_true", help="deterministic autoencoder (no KL, no sampling)")
    ap.add_argument("--beta", type=float, default=1e-6, help="KL weight (ignored with --ae)")
    ap.add_argument("--edm-weight", type=float, default=1e-2,
                    help="rank-3 EDM validity penalty weight (applied only with --decoder matrix)")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of train held out for model selection; 0 = train on everything (PoC / overfit)")
    ap.add_argument("--tag", type=str, default="", help="optional suffix appended to the run name")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 <= args.val_frac < 1:
        ap.error("epochs/batch-size must be positive and val-frac must be in [0, 1)")
    try:
        DEVICE = resolve_device(args.device)
    except ValueError as exc:
        ap.error(str(exc))

    set_seed(args.seed)
    name = run_name(args)
    print(f"run: {name}   device: {DEVICE}")

    train = Shapes3DTensors("train")
    x_all, d_all, p_all = make_inputs(train, args.features)
    channels = x_all.shape[1]

    # optional held-out split from train, for model selection (val-frac 0 -> train on everything)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(x_all))
    n_val = int(len(x_all) * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    pin = DEVICE.startswith("cuda")
    tr_loader = DataLoader(TensorDataset(x_all[tr_idx], d_all[tr_idx], p_all[tr_idx]),
                           batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader = (DataLoader(TensorDataset(x_all[val_idx], d_all[val_idx], p_all[val_idx]),
                             batch_size=256, pin_memory=pin) if n_val > 0 else None)

    model = build_model(args.encoder, channels, train.D_e.shape[-1], args.latent_dim,
                        variational=not args.ae, decoder=args.decoder).to(DEVICE)
    use_edm = args.decoder == "matrix"
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"channels={channels}  params={n_params:,}")

    hist = {"args": vars(args), "n_params": n_params, "train_loss": [], "train_recon": [],
            "val_recon": [], "edm": []}
    best_val, best_state = float("inf"), None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        agg = {"loss": 0.0, "recon": 0.0, "edm": 0.0, "n": 0}
        for x, dtgt, ptgt in tr_loader:
            x, dtgt, ptgt = x.to(DEVICE), dtgt.to(DEVICE), ptgt.to(DEVICE)
            out = model(x)
            rec = _recon_metric(out, dtgt, ptgt)
            edm = edm_rank3_penalty(out["D_hat"]) if use_edm else torch.zeros((), device=DEVICE)
            kl = kl_loss(out["mu"], out["logvar"]) if not args.ae else torch.zeros((), device=DEVICE)
            loss = rec + (args.edm_weight * edm if use_edm else 0.0) + (0.0 if args.ae else args.beta * kl)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = len(x)
            agg["loss"] += loss.item() * bs; agg["recon"] += rec.item() * bs
            agg["edm"] += edm.item() * bs; agg["n"] += bs
        train_recon = agg["recon"] / agg["n"]
        sel_recon = evaluate_recon(model, val_loader) if val_loader is not None else train_recon
        hist["train_loss"].append(agg["loss"] / agg["n"]); hist["train_recon"].append(train_recon)
        hist["edm"].append(agg["edm"] / agg["n"]); hist["val_recon"].append(sel_recon)
        if sel_recon <= best_val:
            best_val = sel_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            tag = "val" if val_loader is not None else "trn"
            print(f"  ep {ep:3d}/{args.epochs}  train_recon {train_recon:.4e}  "
                  f"{tag}_recon {sel_recon:.4e}  edm {agg['edm']/agg['n']:.3e}  ({time.time()-t0:.0f}s)")

    torch.save({"state_dict": best_state, "args": vars(args), "channels": channels,
                "n_points": train.D_e.shape[-1], "best_val_recon": best_val},
               CKPT_DIR / f"{name}_best.pt")
    with open(RES_DIR / f"{name}_history.json", "w") as f:
        json.dump(hist, f, indent=2)
    print(f"saved checkpoint -> {CKPT_DIR / f'{name}_best.pt'}   best selection recon {best_val:.4e}")


if __name__ == "__main__":
    main()
