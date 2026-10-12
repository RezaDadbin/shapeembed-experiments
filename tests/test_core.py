"""CPU checks for geometry, model training, and faithful result visualization."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "legacy"), str(ROOT)]
import config
import losses
import models
import preprocess
import report_figures
import visualize
import run
from shapeembed.models.factory import build_model as build_legacy
from shapeembed.train import compute_model_loss
from shapeembed.models.circular_resnet import Normalize


def load_surface(name):
    spec = importlib.util.spec_from_file_location(f"surface_{name}", ROOT / "phase3" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


surface = load_surface("models")
geometry = load_surface("common")
torch.set_num_threads(2)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_normalized_contour_is_invariant_to_similarity(self):
        pts = np.random.randn(32, 2).astype(np.float32)
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
        D, _ = preprocess.contour_to_distance_matrix(pts)
        moved, _ = preprocess.contour_to_distance_matrix(5 * pts @ rotation + 12)
        np.testing.assert_allclose(D, moved, atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(D)), 1.0, places=6)

    def test_reconstruction_loss_handles_cyclic_shift_and_reversal(self):
        D = torch.cdist(torch.randn(2, 12, 2), torch.randn(2, 12, 2)).unsqueeze(1)
        for direction in (1, -1):
            idx = (torch.arange(12) * direction + 5) % 12
            transformed = D[:, :, idx][:, :, :, idx]
            self.assertEqual(losses.indexation_invariant_reconstruction_loss(transformed, D).item(), 0)
        bad = D.clone()
        bad[:, :, 0, 1] += 2
        self.assertGreater(losses.indexation_invariant_reconstruction_loss(bad, D).item(), 0)

    def test_set_encoder_is_invariant_to_arbitrary_permutation(self):
        points = torch.randn(3, 16, 3)
        D = torch.cdist(points, points)
        D = D / D.norm(dim=(-1, -2), keepdim=True)
        x = torch.stack([D, D.square()], dim=1)
        model = surface.SetEncoder(2, 16, 8, hidden=32).eval()
        p = torch.randperm(16)
        with torch.no_grad():
            a, b = model(x)[0], model(x[:, :, p][:, :, :, p])[0]
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_sorted_loss_is_permutation_invariant(self):
        points = torch.randn(2, 16, 3)
        D = torch.cdist(points, points)
        p = torch.randperm(16)
        self.assertEqual(surface.sorted_recon_loss(D[:, p][:, :, p], D).item(), 0)

    def test_point_decoder_produces_valid_3d_geometry(self):
        decoder = surface.PointCloudDecoder(8, 16, hidden=32)
        out = decoder(torch.randn(2, 8))
        D = out["D_hat"]
        torch.testing.assert_close(D, torch.cdist(out["P_hat"], out["P_hat"]), atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(D, D.transpose(1, 2))
        torch.testing.assert_close(D.norm(dim=(-1, -2)), torch.ones(2))
        self.assertLess(surface.edm_rank3_penalty(D).item(), 1e-8)

    def test_kabsch_recovers_rigid_motion(self):
        points = np.random.randn(30, 3)
        Q, _ = np.linalg.qr(np.random.randn(3, 3))
        Q[:, 0] *= np.linalg.det(Q)
        target = points @ Q + np.array([3, -2, 8])
        np.testing.assert_allclose(geometry.kabsch(points, target), target, atol=1e-10)

    def test_surface_outputs_stay_inside_repository(self):
        self.assertEqual(geometry.ROOT, ROOT)
        self.assertTrue(geometry.DATA_DIR.is_relative_to(ROOT))
        self.assertTrue(geometry.OUT_DIR.is_relative_to(ROOT))


class ModelTests(unittest.TestCase):
    def test_legacy_normalization_handles_blank_input(self):
        output = Normalize()(torch.zeros(2, 1, 32, 32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output.count_nonzero().item(), 0)

    def test_all_main_variants_have_finite_training_gradients(self):
        for kind in ("original", "ae", "mlp_decoder"):
            with self.subTest(kind=kind):
                torch.manual_seed(42)
                model = models.build_model(kind, latent_dim=8, matrix_size=32)
                x = torch.cdist(torch.randn(2, 32, 2), torch.randn(2, 32, 2)).unsqueeze(1)
                out = model(x)
                self.assertEqual(out["recon"].shape, x.shape)
                loss, _ = losses.compute_loss(kind, out)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                torch.optim.SGD(model.parameters(), lr=1e-3).step()
                del model, out, loss

    def test_vae_eval_is_deterministic_and_reflection_invariant(self):
        model = models.build_model("original", latent_dim=8, matrix_size=32).eval()
        x = torch.rand(2, 1, 32, 32)
        with torch.no_grad():
            a, b = model(x), model(x)
            reflected = model(torch.flip(x, (-2, -1)))
        torch.testing.assert_close(a["recon"], b["recon"], atol=0, rtol=0)
        torch.testing.assert_close(a["mu"], reflected["mu"], atol=1e-6, rtol=1e-5)

    def test_legacy_vae_and_ae_can_backpropagate(self):
        for cfg in (config.LEGACY_MNIST_VAE_CONFIG, config.LEGACY_MNIST_AE_CONFIG):
            cfg = copy.deepcopy(cfg)
            cfg["model"]["latent_dim"] = 8
            model = build_legacy(cfg)
            weights = dict(beta=1e-10, gamma=1e-5, delta=1e-5, epsilon=1e-5,
                           reconstruction_loss="mse", indexation_invariant_loss=True,
                           reflection_invariant_loss=True)
            D = torch.rand(2, 1, 32, 32)
            result = compute_model_loss(model, cfg["model"]["type"], D, weights)
            result["total"].backward()
            self.assertTrue(torch.isfinite(result["total"]))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_surface_model_can_train_one_step(self):
        model = surface.build_model(channels=2, n_points=16, latent_dim=8)
        P = torch.randn(2, 16, 3)
        D = torch.cdist(P, P)
        P = P / D.norm(dim=(-1, -2), keepdim=True)
        D = torch.cdist(P, P)
        out = model(torch.stack([D, D.square()], dim=1))
        loss = surface.sorted_recon_loss(out["D_hat"], D)
        loss = loss + 0.05 * surface.aligned_point_loss(out["P_hat"], P)
        loss = loss + 1e-6 * surface.kl_loss(out["mu"], out["logvar"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        torch.optim.Adam(model.parameters(), lr=1e-3).step()


class PresentationTests(unittest.TestCase):
    def test_reconstruction_selection_preserves_raw_decoder_outputs(self):
        class Dataset:
            def __len__(self): return 12
            def __getitem__(self, i):
                return torch.full((1, 4, 4), i / 10), torch.tensor(i % 2)

        class Model(torch.nn.Module):
            def forward(self, x):
                return {"x_pre": x, "recon": x.square() + 7}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            torch.save(torch.zeros(12, 4, 2), path / "contours_test.pt")
            with patch.object(report_figures, "DistanceMatrixDataset", return_value=Dataset()), \
                 patch.object(report_figures, "DATA_PROCESSED", path), \
                 patch.object(report_figures, "DEVICE", "cpu"):
                a, b, _ = report_figures.select_reconstruction_examples(Model(), model_type="original")
                a2, b2, _ = report_figures.select_reconstruction_examples(Model(), model_type="ae")
            np.testing.assert_allclose(b, a ** 2 + 7)
            np.testing.assert_array_equal(a, a2)
            np.testing.assert_array_equal(b, b2)

    def test_tsne_keeps_overlapping_classes(self):
        class Projection:
            def __init__(self, **kwargs): pass
            def fit_transform(self, x): return np.zeros((len(x), 2))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tsne.png"
            with patch.object(visualize, "TSNE", Projection):
                visualize.save_tsne_figure(np.ones((500, 8)), np.arange(500) % 2, path)
            metadata = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(metadata["indices"], list(range(500)))
            self.assertEqual(len(metadata["coordinates"]), 500)
            self.assertTrue(path.is_file())

    def test_device_auto_is_resolved_before_dispatch(self):
        args = ["run.py", "--dataset", "bbbc010", "--model", "vae", "--mode", "train",
                "--device", "auto", "--epochs", "1", "--batch-size", "2"]
        with patch.object(sys, "argv", args), patch("utils.resolve_device", return_value="cpu"), \
             patch.object(run, "run_main") as dispatch:
            run.main()
        dispatch.assert_called_once_with("bbbc010", "vae", "train", "cpu",
                                         epochs=1, batch_size=2, num_workers=None)


if __name__ == "__main__":
    unittest.main()
