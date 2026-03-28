"""
brain_mri_3d.py
===============
PROPER 3D Brain MRI Reconstruction from NIfTI Data

PIPELINE:
  1. Load NIfTI volume (MNI152 or patient scan)
  2. Percentile normalization [0, 1]
  3. Brain segmentation via Otsu thresholding
  4. Morphological skull-stripping (erode → LCC → dilate → fill)
  5. Light Gaussian smoothing (σ=0.7)
  6. Marching cubes surface extraction (step_size=2 for performance)
  7. Sparse-matrix Laplacian mesh smoothing (10 iterations)
  8. Multi-view PNG, rotation GIF, interactive Plotly HTML

Works on Windows — no GPU / no PyVista needed.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_erosion,
    binary_dilation,
    label,
    generate_binary_structure,
)
from scipy.sparse import lil_matrix, csr_matrix
from skimage.measure import marching_cubes
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from tqdm import tqdm
import imageio
import time

# ── PATHS ────────────────────────────────────────────────────
NII_PATH = Path("dataset/brain.nii.gz")
OUT_DIR  = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD NIfTI VOLUME
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1 — Load NIfTI MRI Volume")
print("=" * 60)

if not NII_PATH.exists():
    print("  Downloading MNI152 T1 1 mm template …")
    from nilearn.datasets import load_mni152_template
    img = load_mni152_template(resolution=1)
    NII_PATH.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(NII_PATH))
    print("  Saved → dataset/brain.nii.gz")

img = nib.load(str(NII_PATH))
raw = img.get_fdata(dtype=np.float32)
print(f"  Shape     : {raw.shape}")
print(f"  Voxel size: {img.header.get_zooms()}")
print(f"  Range     : {raw.min():.1f} – {raw.max():.1f}")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — NORMALIZE + BRAIN SEGMENTATION + SKULL STRIP
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2 — Brain Segmentation & Skull Stripping")
print("=" * 60)
t0 = time.time()

# 2a. Robust percentile normalization ─────────────────────────
bg_thresh = raw.max() * 0.03
tissue_voxels = raw[raw > bg_thresh]
p1, p99 = np.percentile(tissue_voxels, [1, 99])
data = np.clip(raw, 0, p99)
data = (data - data.min()) / (data.max() - data.min() + 1e-8)
print(f"  Normalized to [0, 1]  (clipped at p99 = {p99:.1f})")

# 2b. Pre-smoothing for cleaner segmentation ──────────────────
data_smooth = gaussian_filter(data, sigma=0.5)

# 2c. Otsu auto-threshold ─────────────────────────────────────
tissue_mask = data_smooth > 0.05
tissue_vals = data_smooth[tissue_mask]
otsu_val = threshold_otsu(tissue_vals)
seg_threshold = otsu_val * 0.55
print(f"  Otsu threshold: {otsu_val:.3f}  →  seg threshold: {seg_threshold:.3f}")

mask = (data_smooth > seg_threshold).astype(np.uint8)

# 2d. Morphological brain extraction ──────────────────────────
mask = binary_fill_holes(mask).astype(np.uint8)

struct_tight = generate_binary_structure(3, 1)   # 6-connected
struct_wide  = generate_binary_structure(3, 3)   # 26-connected

# Erode to detach skull from brain
mask_erode = binary_erosion(mask, structure=struct_tight, iterations=3).astype(np.uint8)

# Keep largest connected component = brain
lbl_array, n_components = label(mask_erode)
if n_components > 0:
    sizes = np.bincount(lbl_array.ravel())
    sizes[0] = 0
    brain_label = sizes.argmax()
    mask_brain = (lbl_array == brain_label).astype(np.uint8)
    print(f"  Components: {n_components}, kept label={brain_label}")
else:
    mask_brain = mask_erode

# Dilate back to recover cortical surface
mask_brain = binary_dilation(mask_brain, structure=struct_wide, iterations=3).astype(np.uint8)
mask_brain = binary_fill_holes(mask_brain).astype(np.uint8)

brain_pct = 100 * mask_brain.sum() / mask_brain.size
print(f"  Brain mask: {mask_brain.sum():,} voxels ({brain_pct:.1f}%)")

# 2e. Apply mask + smooth ─────────────────────────────────────
vol_brain = data * mask_brain.astype(np.float32)
vol_brain = gaussian_filter(vol_brain, sigma=0.7)
vol_brain = np.clip(vol_brain, 0, 1)

print(f"  Volume: {vol_brain.shape}, range [{vol_brain.min():.3f}, {vol_brain.max():.3f}]")
print(f"  STEP 2 done in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — ORTHOGONAL VIEWS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3 — Orthogonal Slice Views")
print("=" * 60)

D, H, W = vol_brain.shape

ax_sl = np.rot90(vol_brain[:, :, W // 2], k=1)
co_sl = np.rot90(vol_brain[:, H // 2, :], k=1)
sa_sl = np.rot90(vol_brain[D // 2, :, :], k=1)

fig = plt.figure(figsize=(22, 14), facecolor="#050810")
gs  = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.25,
                        left=0.04, right=0.96, top=0.91, bottom=0.04)


def ortho_panel(ax, img_data, title, xl, yl, cmap="gray"):
    ax.set_facecolor("#050810")
    im = ax.imshow(img_data, cmap=cmap, aspect="equal",
                   norm=Normalize(0, 1), interpolation="bilinear")
    ax.set_title(title, color="white", fontsize=13,
                 fontweight="bold", pad=8)
    ax.set_xlabel(xl, color="#aaa", fontsize=9)
    ax.set_ylabel(yl, color="#aaa", fontsize=9)
    ax.tick_params(colors="#444", labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#222")
    cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="w", labelcolor="w", labelsize=7)


ortho_panel(fig.add_subplot(gs[0, 0]), sa_sl,
            f"AXIAL z={D//2}", "L ↔ R", "A ↔ P")
ortho_panel(fig.add_subplot(gs[0, 1]), co_sl,
            f"CORONAL y={H//2}", "L ↔ R", "S ↔ I")
ortho_panel(fig.add_subplot(gs[0, 2]), ax_sl,
            f"SAGITTAL x={W//2}", "A ↔ P", "S ↔ I")
ortho_panel(fig.add_subplot(gs[1, 0]), sa_sl,
            "AXIAL (turbo)", "L ↔ R", "A ↔ P", "turbo")
ortho_panel(fig.add_subplot(gs[1, 1]), co_sl,
            "CORONAL (turbo)", "L ↔ R", "S ↔ I", "turbo")
ortho_panel(fig.add_subplot(gs[1, 2]), ax_sl,
            "SAGITTAL (turbo)", "A ↔ P", "S ↔ I", "turbo")

fig.suptitle("Brain MRI — Skull-Stripped — Orthogonal Views",
             color="white", fontsize=15, fontweight="bold", y=0.97)
fig.savefig(OUT_DIR / "orthogonal_views.png", dpi=150,
            bbox_inches="tight", facecolor="#050810")
plt.close(fig)
print("  ✅ orthogonal_views.png")

# ═══════════════════════════════════════════════════════════════
# STEP 4 — SURFACE EXTRACTION (Marching Cubes)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4 — Marching Cubes Surface Extraction")
print("=" * 60)
t0 = time.time()

# Auto iso-level from percentile of brain voxels
brain_vals = vol_brain[vol_brain > 0.01]
iso_level = np.percentile(brain_vals, 35)
iso_level = np.clip(iso_level, 0.15, 0.45)
print(f"  Iso-level: {iso_level:.3f}")

# step_size=2 for manageable mesh (8x fewer faces than step_size=1)
verts, faces, normals, _ = marching_cubes(vol_brain, level=iso_level, step_size=2)
print(f"  Raw mesh: {len(verts):,} verts, {len(faces):,} faces")

# Center at origin
verts -= verts.mean(axis=0)

# ── Vectorized Laplacian Smoothing (scipy sparse) ────────────
print("  Laplacian smoothing (10 iterations, λ=0.5) …")


def laplacian_smooth_sparse(vertices, triangles, iterations=10, lam=0.5):
    """Vectorized Laplacian smoothing using sparse adjacency matrix."""
    n = len(vertices)

    # Build sparse adjacency matrix
    rows = np.concatenate([triangles[:, 0], triangles[:, 1], triangles[:, 2],
                           triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    cols = np.concatenate([triangles[:, 1], triangles[:, 2], triangles[:, 0],
                           triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    data_vals = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data_vals, (rows, cols)), shape=(n, n))

    # Normalize rows (each row sums to 1 = average of neighbors)
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    inv_sums = 1.0 / row_sums
    # Create diagonal normalization matrix
    from scipy.sparse import diags
    D_inv = diags(inv_sums)
    L = D_inv @ adj   # normalized adjacency

    v = vertices.copy()
    for i in range(iterations):
        avg = L @ v
        v = v + lam * (avg - v)

    return v


verts = laplacian_smooth_sparse(verts, faces, iterations=10, lam=0.5)
print(f"  Smoothed: {len(verts):,} verts, {len(faces):,} faces")
print(f"  STEP 4 done in {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — 4-VIEW PNG RENDER
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 5 — 4-View Clinical PNG")
print("=" * 60)
t0 = time.time()

# Subsample faces for matplotlib rendering (every 3rd for speed)
faces_render = faces[::3]

# Per-face Phong shading
lights = [
    (np.array([ 0.30, -0.75,  0.60]), 1.10),
    (np.array([-0.70,  0.30,  0.20]), 0.50),
    (np.array([ 0.00,  0.70, -0.30]), 0.25),
]
for ld, _ in lights:
    ld /= np.linalg.norm(ld)

v0 = verts[faces_render[:, 0]]
v1 = verts[faces_render[:, 1]]
v2 = verts[faces_render[:, 2]]
fn = np.cross(v1 - v0, v2 - v0)
fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)

intensity = np.zeros(len(faces_render))
for ld, strength in lights:
    intensity += np.clip(np.dot(fn, ld), 0, 1) * strength
intensity = np.clip(intensity / sum(s for _, s in lights), 0, 1)

# Warm bone / cortex color
r = np.clip(0.95 * intensity + 0.05, 0, 1)
g = np.clip(0.75 * intensity + 0.05, 0, 1)
b = np.clip(0.65 * intensity + 0.05, 0, 1)
face_colors = np.column_stack([r, g, b, np.ones(len(faces_render))])

mx = np.abs(verts).max() * 1.08


def draw_brain(ax, elev, azim, title):
    ax.set_facecolor("#050810")
    poly = Poly3DCollection(
        verts[faces_render],
        facecolors=face_colors,
        edgecolors="none",
        shade=False,
        antialiased=True,
    )
    ax.add_collection3d(poly)
    ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx); ax.set_zlim(-mx, mx)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_title(title, color="white", fontsize=13,
                 fontweight="bold", pad=4)


fig4 = plt.figure(figsize=(20, 17), facecolor="#050810")
fig4.suptitle(
    f"Brain MRI — 3D Surface Reconstruction\n"
    f"Skull-Stripped · Marching Cubes (level={iso_level:.2f}) · "
    f"Laplacian Smoothed",
    color="white", fontsize=15, fontweight="bold", y=0.98,
)

draw_brain(fig4.add_subplot(2, 2, 1, projection="3d",
           facecolor="#050810"), 15, -90, "Anterior View")
draw_brain(fig4.add_subplot(2, 2, 2, projection="3d",
           facecolor="#050810"), 88, -90, "Superior View")
draw_brain(fig4.add_subplot(2, 2, 3, projection="3d",
           facecolor="#050810"), 15, 175, "Left Lateral")
draw_brain(fig4.add_subplot(2, 2, 4, projection="3d",
           facecolor="#050810"), 15, -5, "Right Lateral")

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig4.savefig(OUT_DIR / "brain_3d_surface.png", dpi=150,
             bbox_inches="tight", facecolor="#050810")
plt.close(fig4)
print(f"  ✅ brain_3d_surface.png  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — ROTATION GIF
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 6 — Rotation GIF (36 frames @ 12 fps)")
print("=" * 60)
t0 = time.time()

gif_frames = []
for azim in tqdm(np.linspace(0, 360, 36, endpoint=False),
                 desc="  GIF", unit="frame"):
    fig_g = plt.figure(figsize=(7, 6), facecolor="#050810")
    ax_g  = fig_g.add_subplot(111, projection="3d", facecolor="#050810")
    draw_brain(ax_g, elev=22, azim=azim, title="Brain MRI — 3D Surface")
    fig_g.tight_layout()
    fig_g.canvas.draw()

    try:
        buf = fig_g.canvas.buffer_rgba()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(
            fig_g.canvas.get_width_height()[::-1] + (4,)
        )[..., :3]
    except AttributeError:
        frame = np.frombuffer(
            fig_g.canvas.tostring_rgb(), dtype=np.uint8
        ).reshape(fig_g.canvas.get_width_height()[::-1] + (3,))

    gif_frames.append(frame.copy())
    plt.close(fig_g)

imageio.mimsave(str(OUT_DIR / "brain_rotation.gif"),
                gif_frames, fps=12, loop=0)
print(f"  ✅ brain_rotation.gif  ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — PLOTLY INTERACTIVE HTML
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 7 — Plotly Interactive HTML")
print("=" * 60)

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i_f, j_f, k_f = faces[:, 0], faces[:, 1], faces[:, 2]

    vertex_color = (z - z.min()) / (z.max() - z.min() + 1e-8)

    fig_p = go.Figure(data=[go.Mesh3d(
        x=x, y=y, z=z,
        i=i_f, j=j_f, k=k_f,
        intensity=vertex_color,
        colorscale="turbo",
        opacity=1.0,
        flatshading=False,
        showscale=False,
        lighting=dict(
            ambient=0.40, diffuse=0.80,
            specular=0.30, roughness=0.50,
            fresnel=0.12,
        ),
        lightposition=dict(x=150, y=-250, z=350),
        name="Brain Surface",
    )])

    fig_p.update_layout(
        title=dict(
            text=(
                "3D Brain Reconstruction<br>"
                "<span style='font-size:12px;color:#888;'>"
                f"Marching Cubes (level={iso_level:.2f}) · "
                "Laplacian Smoothed · Skull-Stripped"
                "</span>"
            ),
            font=dict(color="white", size=20, family="Outfit, Arial"),
            x=0.5,
        ),
        paper_bgcolor="#050810",
        scene=dict(
            bgcolor="#050810",
            xaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title=""),
            camera=dict(
                eye=dict(x=0, y=-2.2, z=0.5),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=100, b=0),
        annotations=[
            dict(
                text=(
                    f"Vertices: {len(verts):,} · "
                    f"Faces: {len(faces):,}<br>"
                    "Otsu → Morphological Skull-Strip → Laplacian Smooth"
                ),
                showarrow=False, xref="paper", yref="paper",
                x=0.02, y=0.02, align="left",
                font=dict(color="#888888", size=10),
            ),
            dict(
                text=(
                    "Interactive Viewer<br>"
                    "Drag = Rotate · Scroll = Zoom · Shift+Drag = Pan"
                ),
                showarrow=False, xref="paper", yref="paper",
                x=0.98, y=0.02, align="right",
                font=dict(color="#5DC963", size=11),
            ),
        ],
    )

    pyo.plot(fig_p,
             filename=str(OUT_DIR / "brain_3d_interactive.html"),
             auto_open=False,
             include_plotlyjs="cdn")
    print("  ✅ brain_3d_interactive.html")

except ImportError:
    print("  ⚠ pip install plotly  →  then re-run")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ALL OUTPUTS GENERATED:")
print("=" * 60)
for f in sorted(OUT_DIR.iterdir()):
    if f.is_file():
        sz = f.stat().st_size
        v, u = (sz / 1024 / 1024, "MB") if sz > 1e6 else (sz / 1024, "KB")
        print(f"  [{f.suffix[1:].upper():>4}]  {f.name:<42} {v:6.1f} {u}")

print("""
Pipeline Summary:
  ✅ NIfTI loaded & percentile-normalized
  ✅ Otsu auto-threshold for brain segmentation
  ✅ Morphological skull stripping (erode 3x → LCC → dilate 3x → fill)
  ✅ Gaussian smoothing (σ=0.7) on brain volume
  ✅ Marching cubes (step_size=2, auto iso-level)
  ✅ Sparse Laplacian mesh smoothing (10 iterations, λ=0.5)
  ✅ 4-view PNG + rotation GIF + interactive HTML
""")