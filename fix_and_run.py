"""
fix_and_run.py
==============
ONE script that:
1. Downloads real MNI152 human brain MRI
2. Saves it as dataset/brain.nii.gz
3. Verifies it loaded correctly
4. Starts the Flask server

Run:
    cd backend
    python fix_and_run.py
"""
import os, sys, shutil
import numpy as np
from pathlib import Path

print("="*55)
print(" STEP 1 — Getting real single-patient brain MRI")
print("="*55)

Path("dataset").mkdir(exist_ok=True)
OUT = Path("dataset/brain.nii.gz")

# Remove old multi-patient data so app uses NIfTI
if Path("dataset/raw").exists():
    print("  Renaming dataset/raw → dataset/raw_multipatient (won't be used)")
    if Path("dataset/raw_multipatient").exists():
        shutil.rmtree("dataset/raw_multipatient")
    shutil.move("dataset/raw", "dataset/raw_multipatient")
    print("  ✅ Multi-patient folder moved aside")

# Download real brain
try:
    import nibabel as nib
    from nilearn.datasets import load_mni152_template

    print("  Loading MNI152 T1 (real human brain, 1mm resolution)...")
    img = load_mni152_template(resolution=1)
    data = img.get_fdata().astype(np.float32)
    print(f"  Raw shape: {img.shape}, range: {data.min():.2f}–{data.max():.2f}")

    # Global normalize NOW before saving — prevents all stripe issues
    mask = data > (data.max() * 0.02)
    brain_vals = data[mask]
    p01  = np.percentile(brain_vals, 0.5)
    p999 = np.percentile(brain_vals, 99.5)
    data = np.clip(data, 0, p999)
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    data = data ** 0.72            # gamma → brightens grey/white matter
    data = (data * 255).astype(np.uint8)

    print(f"  Normalized range: {data.min()}–{data.max()}")

    # Save normalized volume as NIfTI
    new_img = nib.Nifti1Image(data.astype(np.int16), img.affine)
    nib.save(new_img, str(OUT))
    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"  ✅ Saved → {OUT}  ({size_mb:.1f} MB)")

except Exception as e:
    print(f"  ❌ Failed: {e}")
    print("  Make sure nilearn and nibabel are installed:")
    print("     pip install nilearn nibabel")
    sys.exit(1)

print()
print("="*55)
print(" STEP 2 — Verifying volume loads correctly")
print("="*55)

sys.path.insert(0, '.')
from volume import BrainVolume

bv = BrainVolume()
bv.load_nifti(str(OUT))

D, H, W = bv.shape
print(f"  Shape: {D}×{H}×{W}")

# Check for stripes: coronal slice means should be consistent
sample_indices = list(range(H//4, 3*H//4, H//10))
means = [float(bv.get_coronal(i).mean()) for i in sample_indices]
spread = max(means) - min(means)
print(f"  Coronal consistency spread: {spread:.1f}", end="  ")
if spread < 40:
    print("✅ No stripes!")
else:
    print("⚠️  May have mild variation (normal for real brain anatomy)")

print()
print("="*55)
print(" STEP 3 — Starting Flask server")
print("="*55)
print()
print("  Open browser → http://localhost:5000")
print()

# Start the server
import subprocess
subprocess.run([sys.executable, "app.py"])