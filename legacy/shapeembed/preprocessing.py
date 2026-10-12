from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.interpolate import splev, splprep
from skimage import measure
from skimage.color import rgb2gray


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.dtype == bool:
        return mask

    return mask > 0


def remove_duplicate_endpoint(points: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)

    if len(points) >= 2 and np.allclose(points[0], points[-1], atol=atol):
        points = points[:-1]

    return points


def polygon_signed_area_cartesian(points_xy: np.ndarray) -> float:
    """
    points_xy are image coordinates:
        x = column
        y = row, increasing downward

    For normal Cartesian orientation, we flip y.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)

    x = points_xy[:, 0]
    y = -points_xy[:, 1]

    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    area = 0.5 * np.sum(x * y_next - x_next * y)
    return float(area)


def ensure_counterclockwise(points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)

    area = polygon_signed_area_cartesian(points_xy)

    if area < 0:
        points_xy = points_xy[::-1].copy()

    return points_xy


def contour_area_abs(points_xy: np.ndarray) -> float:
    return abs(polygon_signed_area_cartesian(points_xy))


def extract_largest_contour(mask: np.ndarray) -> np.ndarray:
    """
    Extract the longest valid marching-squares contour.

    Returns:
        points_xy: [M, 2], stored as (x, y_image)
    """
    mask_bool = binarize_mask(mask)

    contours = measure.find_contours(mask_bool.astype(np.float32), level=0.5)
    valid_contours = [contour for contour in contours if len(contour) >= 4]

    if len(valid_contours) == 0:
        raise ValueError("No contour found in mask.")

    contour_rc = max(valid_contours, key=len)

    # skimage returns (row, col). Convert to (x, y).
    best_points = np.stack(
        [contour_rc[:, 1], contour_rc[:, 0]],
        axis=1,
    ).astype(np.float64)

    best_points = remove_duplicate_endpoint(best_points)

    if len(best_points) < 4:
        raise ValueError("No valid contour found in mask.")

    return best_points.astype(np.float64)


def extract_official_contour(mask: np.ndarray) -> np.ndarray:
    """
    Official ConvertBinaryMasksToDMs-style contour extraction:
        - no area sorting
        - no orientation correction
        - no duplicate-endpoint removal here
        - choose the longest contour
        - skimage.find_contours(mask, 0.8)
    """
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = rgb2gray(mask)

    mask = mask.astype(np.float32)

    # Important for torchvision MNIST, whose values are usually 0..255.
    # Official helper assumes binary-like masks around 0/1.
    if mask.max() > 1.0:
        mask = mask / 255.0

    contours = measure.find_contours(mask, 0.8)
    contours = sorted(contours, key=lambda x: len(x), reverse=True)

    if len(contours) == 0:
        raise ValueError("No contour found in mask.")

    contour = contours[0]
    x, y = contour[:, 0], contour[:, 1]

    return np.column_stack([x, y]).astype(np.float64)


def extract_contour_points_from_mask(
    mask: np.ndarray,
    mode: str = "project",
) -> np.ndarray:
    mode = str(mode).lower()

    if mode in ["official", "longest"]:
        return extract_official_contour(mask)

    if mode in ["project", "largest", "largest_area", "area"]:
        return extract_largest_contour(mask)

    raise ValueError(f"Unknown contour_extraction mode: {mode}")


def linear_arc_length_resample(points_xy: np.ndarray, n_points: int) -> np.ndarray:
    """
    Paper-style contour sampling:

    closed contour polyline
    -> cumulative arc length
    -> uniformly sample N positions
    -> linear interpolation

    No smooth spline.
    No scipy splprep/splev.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)

    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"Expected points of shape [M, 2], got {points_xy.shape}")

    if len(points_xy) < 4:
        raise ValueError("Need at least 4 contour points.")

    points_xy = remove_duplicate_endpoint(points_xy)
    points_xy = ensure_counterclockwise(points_xy)

    closed = np.vstack([points_xy, points_xy[0]])
    seg_lengths = np.sqrt(((closed[1:] - closed[:-1]) ** 2).sum(axis=1))
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumlen[-1]

    if total <= 1e-12:
        raise ValueError("Degenerate contour: zero total length.")

    sample_d = np.linspace(
        0.0,
        total,
        n_points,
        endpoint=False,
    )

    x = np.interp(sample_d, cumlen, closed[:, 0])
    y = np.interp(sample_d, cumlen, closed[:, 1])
    sampled = np.stack([x, y], axis=1)

    return sampled.astype(np.float32)


def spline_arc_length_resample(points_xy: np.ndarray, n_points: int) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)

    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"Expected points of shape [M, 2], got {points_xy.shape}")

    points_xy = remove_duplicate_endpoint(points_xy)
    points_xy = ensure_counterclockwise(points_xy)

    if len(points_xy) < 4:
        raise ValueError("Need at least 4 contour points.")

    k = min(3, len(points_xy) - 1)

    tck, _ = splprep(
        [points_xy[:, 0], points_xy[:, 1]],
        s=0.0,
        per=True,
        k=k,
    )

    sample_u = np.linspace(
        0.0,
        1.0,
        num=n_points,
        endpoint=False,
        dtype=np.float64,
    )

    x, y = splev(sample_u, tck)
    sampled = np.stack([x, y], axis=1)
    sampled = ensure_counterclockwise(sampled)

    return sampled.astype(np.float32)


def official_spline_resample(points_xy: np.ndarray, n_points: int) -> np.ndarray:
    """
    Official ConvertBinaryMasksToDMs-style periodic spline resampling.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)

    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"Expected points of shape [M, 2], got {points_xy.shape}")

    if len(points_xy) < 4:
        raise ValueError("Need at least 4 contour points.")

    tck, u = splprep(
        points_xy.T,
        s=0.0,
        k=min(3, len(points_xy) - 1),
        per=True,
    )
    sample_u = np.linspace(u.min(), u.max(), n_points)
    sampled = np.column_stack(splev(sample_u, tck))

    return sampled.astype(np.float32)


def resample_contour_points(
    points_xy: np.ndarray,
    n_points: int,
    mode: str = "linear",
) -> np.ndarray:
    points, _ = resample_contour_points_with_info(
        points_xy,
        n_points=n_points,
        mode=mode,
    )
    return points


def resample_contour_points_with_info(
    points_xy: np.ndarray,
    n_points: int,
    mode: str = "linear",
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(mode).lower()
    info: dict[str, Any] = {
        "contour_resampling": mode,
        "resampling_fallback": None,
        "resampling_error": None,
    }

    if mode == "linear":
        return linear_arc_length_resample(points_xy, n_points=n_points), info

    if mode == "official":
        try:
            return official_spline_resample(points_xy, n_points=n_points), info
        except Exception as exc:
            info["resampling_error"] = str(exc)

            try:
                info["resampling_fallback"] = "spline"
                return spline_arc_length_resample(points_xy, n_points=n_points), info
            except Exception as spline_exc:
                info["resampling_error"] = (
                    f"official={exc}; spline={spline_exc}"
                )
                info["resampling_fallback"] = "linear"
                return linear_arc_length_resample(points_xy, n_points=n_points), info

    if mode != "spline":
        raise ValueError(f"Unknown contour_resampling mode: {mode}")

    try:
        return spline_arc_length_resample(points_xy, n_points=n_points), info
    except Exception as exc:
        info["resampling_fallback"] = "linear"
        info["resampling_error"] = str(exc)
        return linear_arc_length_resample(points_xy, n_points=n_points), info


def mask_to_contour_points(
    mask: np.ndarray,
    n_points: int,
    resampling: str = "linear",
    contour_extraction: str = "project",
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    contour = extract_contour_points_from_mask(mask, mode=contour_extraction)
    points, info = resample_contour_points_with_info(
        contour,
        n_points=n_points,
        mode=resampling,
    )

    info.update(
        {
            "contour_extraction": str(contour_extraction).lower(),
            "contour_level": 0.8
            if str(contour_extraction).lower() in ["official", "longest"]
            else 0.5,
            "num_contour_points": int(len(contour)),
            "num_sampled_points": int(len(points)),
        }
    )

    if return_info:
        return points.astype(np.float32), info

    return points.astype(np.float32)


def pairwise_distance_matrix(points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32)

    diff = points_xy[:, None, :] - points_xy[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=-1))

    return D.astype(np.float32)


def normalized_distance_matrix(points_xy: np.ndarray) -> Tuple[np.ndarray, float]:
    D = pairwise_distance_matrix(points_xy)
    norm = float(np.linalg.norm(D, ord="fro"))

    if norm <= 1e-12:
        raise ValueError("Distance matrix norm is zero.")

    D_normalized = D / norm
    return D_normalized.astype(np.float32), norm


def mask_to_distance_matrix(
    mask: np.ndarray,
    n_points: int,
    resampling: str = "linear",
    contour_extraction: str = "project",
    return_info: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, float]
    | tuple[np.ndarray, np.ndarray, float, dict[str, Any]]
):
    result = mask_to_contour_points(
        mask,
        n_points=n_points,
        resampling=resampling,
        contour_extraction=contour_extraction,
        return_info=return_info,
    )

    if return_info:
        points, info = result
    else:
        points = result
        info = {}

    D, norm = normalized_distance_matrix(points)

    if return_info:
        return points, D, norm, info

    return points, D, norm


# Backward-compatible aliases.

def extract_contour_points(mask: np.ndarray, n_points: int) -> np.ndarray:
    return mask_to_contour_points(mask, n_points=n_points)


def resample_contour(points_xy: np.ndarray, n_points: int) -> np.ndarray:
    return linear_arc_length_resample(points_xy, n_points=n_points)


def resample_closed_contour(points_xy: np.ndarray, n_points: int) -> np.ndarray:
    return linear_arc_length_resample(points_xy, n_points=n_points)


def preprocess_mask(mask: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray, float]:
    return mask_to_distance_matrix(mask, n_points=n_points)

def rgb2grey(rgb, cr=0.2989, cg=0.5870, cb=0.1140):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return cr * r + cg * g + cb * b
