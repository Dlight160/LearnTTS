"""
判别器 (MSD + MPD)
===================
直接从 stage2_hifi_gan 复用，避免重复定义。
"""

import sys
import importlib.util
from pathlib import Path


def _import_from_stage2(name: str):
    """从 stage2_hifi_gan.model 导入模块。"""
    import importlib
    spec = importlib.util.spec_from_file_location(
        "stage2_model",
        str(Path(__file__).parent.parent / "stage2_hifi_gan" / "model.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


MultiScaleDiscriminator = _import_from_stage2("MultiScaleDiscriminator")
MultiPeriodDiscriminator = _import_from_stage2("MultiPeriodDiscriminator")


if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 1, 16384).to(device)
    msd = MultiScaleDiscriminator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)

    msd_out = msd(x)
    mpd_out = mpd(x)
    print(f"MSD: {len(msd_out)} scales, first logit shape: {msd_out[0][1].shape}")
    print(f"MPD: {len(mpd_out)} periods, first logit shape: {mpd_out[0][1].shape}")
    print("✓ 判别器导入成功")