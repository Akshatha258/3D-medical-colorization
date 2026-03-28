"""
colorizer.py — Brain MRI colorizer (U-Net + colormap fallback)
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from skimage import color as skcolor
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = F = None


def _build_unet_classes():
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class EncoderBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = DoubleConv(in_ch, out_ch)
            self.pool = nn.MaxPool2d(2, 2)
        def forward(self, x):
            skip = self.conv(x)
            return self.pool(skip), skip

    class DecoderBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)
        def forward(self, x, skip):
            x = self.up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            return self.conv(torch.cat([x, skip], dim=1))

    class ColorizationUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = EncoderBlock(1,    64)
            self.enc2 = EncoderBlock(64,  128)
            self.enc3 = EncoderBlock(128, 256)
            self.enc4 = EncoderBlock(256, 512)
            self.bottleneck = nn.Sequential(DoubleConv(512, 1024), nn.Dropout2d(0.25))
            self.dec4 = DecoderBlock(1024, 512, 512)
            self.dec3 = DecoderBlock( 512, 256, 256)
            self.dec2 = DecoderBlock( 256, 128, 128)
            self.dec1 = DecoderBlock( 128,  64,  64)
            self.head = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32,  2, 1), nn.Tanh()
            )
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None: nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

        def forward(self, x):
            x1, s1 = self.enc1(x)
            x2, s2 = self.enc2(x1)
            x3, s3 = self.enc3(x2)
            x4, s4 = self.enc4(x3)
            b  = self.bottleneck(x4)
            d  = self.dec4(b,  s4)
            d  = self.dec3(d,  s3)
            d  = self.dec2(d,  s2)
            d  = self.dec1(d,  s1)
            return self.head(d)

    return ColorizationUNet


class BrainColorizer:
    IMAGE_SIZE  = 224
    MODEL_PATHS = [
        "models/colorization_model_best.pth",
        "models/colorization_model_final.pth",
    ]

    def __init__(self, model_path=None):
        self.device = "cpu"
        self.model  = None
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._try_load_model()

    def _try_load_model(self):
        if not TORCH_AVAILABLE:
            print("  ℹ PyTorch not installed → using medical colormap fallback")
            return
        ColorizationUNet = _build_unet_classes()
        for path_str in self.MODEL_PATHS:
            p = Path(path_str)
            if not p.exists():
                continue
            try:
                ckpt  = torch.load(str(p), map_location=self.device)
                state = ckpt.get("model_state_dict", ckpt)
                net   = ColorizationUNet().to(self.device)
                net.load_state_dict(state)
                net.eval()
                self.model = net
                n = sum(param.numel() for param in net.parameters())
                print(f"  ✓ ColorizationUNet loaded from '{p}'  ({n:,} params | {self.device})")
                return
            except Exception as exc:
                print(f"  ⚠ Could not load '{p}': {exc}")
        print("  ℹ No checkpoint found → using medical colormap fallback")
        print("    (Place models/colorization_model_best.pth to enable U-Net)")

    @property
    def has_model(self):
        return self.model is not None

    def colorize_slice(self, gray):
        if self.model is not None:
            return self._unet_colorize(gray)
        return self._colormap_colorize(gray)

    def _unet_colorize(self, gray):
        H, W = gray.shape
        pil  = Image.fromarray(gray).resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.LANCZOS)
        arr  = np.array(pil, dtype=np.float32) / 255.0
        arr  = gaussian_filter(arr, sigma=0.6)
        L_norm = arr * 2.0 - 1.0
        L_t = torch.FloatTensor(L_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ab_pred = self.model(L_t)
        ab     = ab_pred[0].cpu().numpy().transpose(1, 2, 0)
        ab_real = ab * 128.0
        L_real  = (arr + 1.0) * 50.0
        lab = np.concatenate([L_real[:, :, np.newaxis], ab_real], axis=-1)
        rgb = np.clip(skcolor.lab2rgb(lab), 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        if (H, W) != (self.IMAGE_SIZE, self.IMAGE_SIZE):
            rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.LANCZOS))
        return rgb

    def _colormap_colorize(self, gray):
        """
        Jet/Rainbow medical colormap — matches U-Net notebook output style.
        Background: dark purple/black
        CSF/dark:   blue
        Grey matter: cyan → green
        White matter: yellow → orange
        Bright/skull: red → white
        """
        n = gray.astype(np.float32) / 255.0  # [0, 1]

        # Pure jet colormap (matches matplotlib jet exactly)
        # Black for background, then blue→cyan→green→yellow→red→white
        r = np.zeros_like(n)
        g = np.zeros_like(n)
        b = np.zeros_like(n)

        # Background (very dark) → deep purple/black
        m = n < 0.05
        r = np.where(m, n * 1.0,  r)
        g = np.where(m, 0.0,      g)
        b = np.where(m, n * 2.0,  b)

        # 0.05–0.20 → dark blue to blue
        m = (n >= 0.05) & (n < 0.20); t = (n - 0.05) / 0.15
        r = np.where(m, 0.0,              r)
        g = np.where(m, 0.0,              g)
        b = np.where(m, 0.5 + t * 0.5,   b)

        # 0.20–0.35 → blue to cyan
        m = (n >= 0.20) & (n < 0.35); t = (n - 0.20) / 0.15
        r = np.where(m, 0.0,          r)
        g = np.where(m, t * 1.0,      g)
        b = np.where(m, 1.0,          b)

        # 0.35–0.50 → cyan to green
        m = (n >= 0.35) & (n < 0.50); t = (n - 0.35) / 0.15
        r = np.where(m, 0.0,          r)
        g = np.where(m, 1.0,          g)
        b = np.where(m, 1.0 - t,      b)

        # 0.50–0.65 → green to yellow
        m = (n >= 0.50) & (n < 0.65); t = (n - 0.50) / 0.15
        r = np.where(m, t * 1.0,      r)
        g = np.where(m, 1.0,          g)
        b = np.where(m, 0.0,          b)

        # 0.65–0.80 → yellow to red
        m = (n >= 0.65) & (n < 0.80); t = (n - 0.65) / 0.15
        r = np.where(m, 1.0,          r)
        g = np.where(m, 1.0 - t,      g)
        b = np.where(m, 0.0,          b)

        # 0.80–1.00 → red to bright white-red (skull/very bright)
        m = n >= 0.80; t = (n - 0.80) / 0.20
        r = np.where(m, 1.0,          r)
        g = np.where(m, t * 0.8,      g)
        b = np.where(m, t * 0.8,      b)

        return (np.clip(np.stack([r, g, b], axis=-1), 0, 1) * 255).astype(np.uint8)

    def colorize_volume(self, gray_volume, verbose=True):
        D = gray_volume.shape[0]
        it = tqdm(range(D), desc="  Colorizing", unit="slice") if verbose else range(D)
        results = [self.colorize_slice(gray_volume[i]) for i in it]
        return np.stack(results, axis=0)