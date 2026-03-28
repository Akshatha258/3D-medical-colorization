"""
app.py
======
Flask backend for the 3-D Brain MRI Visualization system.

Endpoints
---------
GET /                             → serves frontend/index.html
GET /api/info                     → volume metadata (shape, has_model)
GET /api/slice                    → one orthogonal slice as PNG data-URL
GET /api/volume/preview           → downsampled volume array for 3-D viewer

Query parameters for /api/slice:
  plane : axial | coronal | sagittal
  idx   : integer slice index
  mode  : gray | color

Query parameters for /api/volume/preview:
  d, h, w : target dimensions (default 32, 64, 64)
  mode    : gray | color

Run
---
  cd backend
  python app.py
  → http://localhost:5000
"""

from __future__ import annotations

import io
import os
import base64
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS
from PIL import Image

from volume    import BrainVolume
from colorizer import BrainColorizer

# ─────────────────────────────────────────────────────────────────────────────
# Flask app setup
# ─────────────────────────────────────────────────────────────────────────────

FRONTEND_DIR = str(Path(__file__).parent.parent / "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# Volume + colorizer initialisation  (runs once at startup)
# ─────────────────────────────────────────────────────────────────────────────

def build_volume() -> BrainVolume:
    vol = BrainVolume()
    slice_dir = Path("dataset/raw")

    if slice_dir.exists():
        imgs = sorted([
            p for p in slice_dir.glob("*.*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
        if len(imgs) >= 5:
            print(f"📂 Loading {len(imgs)} slices from: {slice_dir}")
            # Load all images resized to same square size
            import numpy as np
            from PIL import Image as PILImage
            slices = []
            size = 224
            for p in imgs:
                arr = np.array(
                    PILImage.open(p).convert("L").resize((size, size))
                )
                slices.append(arr)
            data = np.stack(slices, axis=0)  # (D, H, W)
            # Normalize full volume together for consistent contrast
            lo, hi = data.min(), data.max()
            data = ((data - lo) / (hi - lo + 1e-8) * 255).astype("uint8")
            vol.gray  = data
            vol.shape = data.shape
            print(f"   Volume shape: {data.shape}")
            return vol

    print("📂 No dataset found → generating synthetic brain volume")
    vol.generate_synthetic(depth=60, hw=128)
    return vol


print("\n" + "═" * 60)
print(" 🧠  3-D Brain MRI Visualisation System — Backend")
print("═" * 60)

print("\n[1/3] Building volume …")
t0 = time.time()
VOLUME = build_volume()
print(f"      Shape: {VOLUME.shape}  ({time.time()-t0:.1f}s)")

print("\n[2/3] Loading colorizer …")
COLORIZER = BrainColorizer(model_path="models")

print("\n[3/3] Colorising volume …")
t0 = time.time()
VOLUME.set_color_volume(COLORIZER.colorize_volume(VOLUME.gray))
print(f"      Done in {time.time()-t0:.1f}s")

print("\n✅  System ready — http://localhost:5000\n" + "═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_SIZE = 320   # max px for slice display


def _to_pil(arr: np.ndarray) -> Image.Image:
    """Convert (H,W) or (H,W,3) uint8 array to PIL Image."""
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _resize_for_display(arr: np.ndarray, max_size: int = DISPLAY_SIZE) -> np.ndarray:
    """
    Resize slice to fit within max_size × max_size while preserving aspect ratio.
    Minimum 64px on any axis.
    """
    pil = _to_pil(arr)
    w, h = pil.size
    scale = min(max_size / max(w, h), 1.0)
    new_w = max(64, int(w * scale))
    new_h = max(64, int(h * scale))
    pil   = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil)


def _array_to_data_url(arr: np.ndarray) -> str:
    """Convert numpy array to PNG data-URL string."""
    pil = _to_pil(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _get_slice(plane: str, idx: int, mode: str) -> np.ndarray:
    """
    Extract one orthogonal slice from the volume.

    Returns either a grayscale (H,W) or color (H,W,3) uint8 array.
    Flips are applied so the image matches standard anatomical orientation.
    """
    use_color = (mode == "color")

    if plane == "axial":
        arr = VOLUME.get_axial_color(idx) if use_color else VOLUME.get_axial(idx)
        # Flip L-R for radiological convention (patient's left on viewer's right)
        arr = np.fliplr(arr)

    elif plane == "coronal":
        arr = VOLUME.get_coronal_color(idx) if use_color else VOLUME.get_coronal(idx)
        # Flip vertical so Superior is at the top
        arr = np.flipud(arr)

    elif plane == "sagittal":
        arr = VOLUME.get_sagittal_color(idx) if use_color else VOLUME.get_sagittal(idx)
        arr = np.flipud(arr)

    else:
        raise ValueError(f"Unknown plane: {plane}")

    return arr


def _plane_max_idx(plane: str) -> int:
    D, H, W = VOLUME.shape
    return {"axial": D - 1, "coronal": H - 1, "sagittal": W - 1}[plane]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend SPA."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/info")
def api_info():
    """Return volume metadata."""
    D, H, W = VOLUME.shape
    return jsonify({
        "depth":     D,
        "height":    H,
        "width":     W,
        "has_model": COLORIZER.has_model,
        "device":    str(COLORIZER.device),
    })


@app.route("/api/slice")
def api_slice():
    """
    Return a single orthogonal slice as a PNG data-URL.

    Query params: plane, idx, mode
    """
    plane = request.args.get("plane", "axial").lower()
    mode  = request.args.get("mode",  "gray").lower()

    if plane not in ("axial", "coronal", "sagittal"):
        return jsonify({"error": f"Invalid plane '{plane}'"}), 400
    if mode not in ("gray", "color"):
        return jsonify({"error": f"Invalid mode '{mode}'"}), 400

    max_idx = _plane_max_idx(plane)
    idx     = int(np.clip(int(request.args.get("idx", max_idx // 2)), 0, max_idx))

    try:
        arr         = _get_slice(plane, idx, mode)
        display_arr = _resize_for_display(arr)
        data_url    = _array_to_data_url(display_arr)

        return jsonify({
            "image":    data_url,
            "plane":    plane,
            "idx":      idx,
            "max_idx":  max_idx,
            "mode":     mode,
            "shape":    list(arr.shape[:2]),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/volume/preview")
def api_volume_preview():
    """
    Return a spatially downsampled 3-D volume as a flat uint8 list.
    Used by the Three.js viewer for interactive volume rendering.

    Query params:
      d, h, w  — target dimensions (default 32, 48, 48)
      mode     — gray | color
    """
    d    = max(8,  min(64, int(request.args.get("d", 32))))
    h    = max(8,  min(96, int(request.args.get("h", 48))))
    w    = max(8,  min(96, int(request.args.get("w", 48))))
    mode = request.args.get("mode", "gray").lower()

    try:
        if mode == "color" and VOLUME.color is not None:
            vol      = VOLUME.get_downsampled_color((d, h, w))   # (d,h,w,3)
            channels = 3
        else:
            vol      = VOLUME.get_downsampled((d, h, w))         # (d,h,w)
            channels = 1

        return jsonify({
            "data":     vol.flatten().tolist(),
            "shape":    [d, h, w],
            "channels": channels,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/volume/slice_batch")
def api_slice_batch():
    """
    Return multiple slices at once (for preloading thumbnail strips).

    Query params:
      plane    — axial | coronal | sagittal
      mode     — gray | color
      count    — number of thumbnail slices to return (default 10)
      size     — thumbnail size px (default 64)
    """
    plane = request.args.get("plane", "axial").lower()
    mode  = request.args.get("mode",  "gray").lower()
    count = max(2, min(30, int(request.args.get("count", 10))))
    size  = max(32, min(128, int(request.args.get("size", 64))))

    if plane not in ("axial", "coronal", "sagittal"):
        return jsonify({"error": "Invalid plane"}), 400

    max_idx = _plane_max_idx(plane)
    indices = [int(round(i * max_idx / (count - 1))) for i in range(count)]

    thumbnails = []
    for idx in indices:
        arr = _get_slice(plane, idx, mode)
        pil = _to_pil(arr).resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        thumbnails.append({
            "idx":   idx,
            "image": f"data:image/jpeg;base64,{b64}",
        })

    return jsonify({"plane": plane, "thumbnails": thumbnails})


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
