"""
volume.py — Brain MRI Volume builder with CLAHE contrast enhancement
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from PIL import Image
from PIL import ImageOps
from scipy.ndimage import gaussian_filter, zoom

try:
    import nibabel as nib
    HAS_NIB = True
except ImportError:
    HAS_NIB = False


def _clahe_slice(arr: np.ndarray) -> np.ndarray:
    """
    Apply contrast enhancement to a single grayscale slice.
    Works on any float range (0-1, 0-255, etc.).
    Makes brain tissue, ventricles and tumors clearly visible.
    """
    arr = arr.astype(np.float32)

    # Scale any range to 0-255 first
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - lo) / (hi - lo) * 255.0

    # Clip extreme outliers using only non-background pixels
    nonzero = arr[arr > 5]
    if len(nonzero) > 100:
        p1  = np.percentile(nonzero, 1)
        p99 = np.percentile(nonzero, 99)
        arr = np.clip(arr, p1, p99)
        # Re-normalize after clipping
        lo2, hi2 = arr.min(), arr.max()
        if hi2 - lo2 > 1e-8:
            arr = (arr - lo2) / (hi2 - lo2) * 255.0

    # Gamma correction: brightens mid-tones (grey matter, white matter)
    arr = (arr / 255.0) ** 0.75 * 255.0

    return np.clip(arr, 0, 255).astype(np.uint8)


class BrainVolume:
    """
    Container for a 3-D MRI brain volume with orthogonal slice extraction.

    Coordinate conventions:
      D = depth    → axial slices   (superior → inferior)
      H = height   → coronal slices (anterior → posterior)
      W = width    → sagittal slices (left → right)
    """

    def __init__(self):
        self.gray:  np.ndarray | None = None   # (D, H, W) uint8
        self.color: np.ndarray | None = None   # (D, H, W, 3) uint8
        self.shape: tuple | None = None

    # ── Loaders ──────────────────────────────────────────────────────────────

    def load_nifti(self, path: str) -> "BrainVolume":
        if not HAS_NIB:
            raise RuntimeError("nibabel is required: pip install nibabel")
        img  = nib.load(path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim == 3:
            data = data.transpose(2, 1, 0)   # (W,H,D) → (D,H,W)
        print(f"  NIfTI shape (D,H,W): {data.shape}")

        # ── GLOBAL normalization — prevents stripes in coronal/sagittal ──────
        # Scale any float range (0-1 or 0-4095 etc.) to 0-255 consistently
        data = data.astype(np.float32)

        # Remove background noise using brain mask threshold
        threshold = data.max() * 0.02
        brain_voxels = data[data > threshold]
        if len(brain_voxels) > 1000:
            p1  = np.percentile(brain_voxels, 1)
            p99 = np.percentile(brain_voxels, 99.5)
            data = np.clip(data, 0, p99)   # clip only top end to preserve structure
            # Global normalize
            lo, hi = data.min(), data.max()
            data = (data - lo) / (hi - lo + 1e-8)
        else:
            # Already 0-1 range
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)

        # Gamma correction: brightens grey matter and white matter details
        data = data ** 0.72

        data = (data * 255).astype(np.uint8)
        self._store(data)
        print(f"  Volume ready — shape: {self.shape}, range: {data.min()}-{data.max()}")
        return self

    def load_directory(self, directory: str, hw: int = 224) -> "BrainVolume":
        """
        Load PNG/JPG slices from folder, applying CLAHE contrast per slice
        so brain anatomy (ventricles, cortex, tumors) is clearly visible.
        """
        exts  = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        paths: list[Path] = []
        for e in exts:
            paths.extend(Path(directory).glob(e))
        paths = sorted(paths, key=lambda p: p.name)

        if not paths:
            raise FileNotFoundError(f"No images in: {directory}")

        print(f"  Loading & enhancing {len(paths)} slices …")
        slices = []
        for p in paths:
            # Load as grayscale
            pil = Image.open(p).convert("L")
            # Resize to square
            pil = pil.resize((hw, hw), Image.LANCZOS)
            arr = np.array(pil, dtype=np.float32)
            # Apply per-slice contrast enhancement
            enhanced = _clahe_slice(arr)
            slices.append(enhanced)

        data = np.stack(slices, axis=0)  # (D, H, W)
        self._store(data)
        print(f"  Volume ready — shape: {self.shape}")
        return self

    def generate_synthetic(self, depth: int = 60, hw: int = 128) -> "BrainVolume":
        D, H, W = depth, hw, hw
        print(f"  Generating synthetic brain volume ({D}×{H}×{W}) …")

        z = np.linspace(-1.0, 1.0, D)
        y = np.linspace(-1.0, 1.0, H)
        x = np.linspace(-1.0, 1.0, W)
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

        outer_r  = (X/0.88)**2 + (Y/0.93)**2 + (Z/0.80)**2
        inner_r  = (X/0.80)**2 + (Y/0.85)**2 + (Z/0.72)**2
        skull    = np.clip(1.0 - outer_r, 0, 1) * np.clip(inner_r - 0.95, 0, 1)
        brain_r  = (X/0.78)**2 + (Y/0.82)**2 + (Z/0.70)**2
        mask     = np.where(brain_r <= 1.0, 1.0, 0.0)
        wm       = np.exp(-((X/0.50)**2 + (Y/0.58)**2 + (Z/0.50)**2) * 2.8)
        gm       = np.exp(-((X/0.70)**2 + (Y/0.78)**2 + (Z/0.66)**2) * 2.2)
        cc       = np.exp(-(X**2/0.18 + Y**2/0.004 + (Z+0.08)**2/0.04) * 12)
        v1       = np.exp(-((X-0.22)**2/0.007 + Y**2/0.022 + Z**2/0.020) * 28)
        v2       = np.exp(-((X+0.22)**2/0.007 + Y**2/0.022 + Z**2/0.020) * 28)
        stem     = np.exp(-(X**2/0.025 + (Y+0.50)**2/0.065 + Z**2/0.025) * 8)
        cereb    = np.exp(-(X**2/0.14 + Y**2/0.07 + (Z-0.58)**2/0.10) * 4.5)

        volume = (skull*1.0 + wm*0.85 + gm*0.55 + cc*0.30
                  - (v1+v2)*1.0 + stem*0.60 + cereb*0.50)

        rng    = np.random.RandomState(42)
        volume += gaussian_filter(rng.normal(0, 0.04, volume.shape), sigma=0.8)
        volume += gaussian_filter(rng.normal(0, 0.02, volume.shape), sigma=6.0)
        volume  = gaussian_filter(volume, sigma=0.7) * mask
        volume  = np.clip(volume, 0, None)

        # Enhance contrast
        enhanced = np.stack([_clahe_slice(volume[i]) for i in range(D)], axis=0)
        self._store(enhanced)
        print(f"  Synthetic volume ready — shape: {self.shape}")
        return self

    # ── Internal ──────────────────────────────────────────────────────────────

    def _store(self, data: np.ndarray):
        self.gray  = data.astype(np.uint8)
        self.shape = self.gray.shape

    # ── Grayscale slices ──────────────────────────────────────────────────────

    def get_axial(self, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.shape[0]-1))
        return self.gray[idx]

    def get_coronal(self, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.shape[1]-1))
        return self.gray[:, idx, :]

    def get_sagittal(self, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.shape[2]-1))
        return self.gray[:, :, idx]

    # ── Color slices ──────────────────────────────────────────────────────────

    def get_axial_color(self, idx: int) -> np.ndarray:
        if self.color is None: raise RuntimeError("No color volume")
        idx = int(np.clip(idx, 0, self.shape[0]-1))
        return self.color[idx]

    def get_coronal_color(self, idx: int) -> np.ndarray:
        if self.color is None: raise RuntimeError("No color volume")
        idx = int(np.clip(idx, 0, self.shape[1]-1))
        return self.color[:, idx, :]

    def get_sagittal_color(self, idx: int) -> np.ndarray:
        if self.color is None: raise RuntimeError("No color volume")
        idx = int(np.clip(idx, 0, self.shape[2]-1))
        return self.color[:, :, idx]

    # ── Downsampled ───────────────────────────────────────────────────────────

    def get_downsampled(self, target=(32, 64, 64)) -> np.ndarray:
        factors = [t/s for t, s in zip(target, self.shape)]
        return zoom(self.gray.astype(np.float32), factors, order=1).astype(np.uint8)

    def get_downsampled_color(self, target=(32, 64, 64)) -> np.ndarray | None:
        if self.color is None: return None
        factors = [t/s for t, s in zip(target, self.shape)] + [1]
        return zoom(self.color.astype(np.float32), factors, order=1).astype(np.uint8)

    # ── Setter ────────────────────────────────────────────────────────────────

    def set_color_volume(self, color: np.ndarray):
        expected = self.shape + (3,)
        if color.shape != expected:
            raise ValueError(f"Expected {expected}, got {color.shape}")
        self.color = color.astype(np.uint8)