"""根因 F 的回归测试：init_device() 不得自动选择 MPS。

DeepLens 使用 float64 进行波传播/相干光线追迹，而 MPS 后端无法表示该类型。因此，
自动选择 MPS 会使所有双精度工作流崩溃（每当构造 float64 镜头时，还会在测试套件
中引发连锁故障）。现在 init_device() 会在 Apple Silicon 上回退到 CPU，而不返回
MPS 设备。
"""

import torch

from deeplens import init_device


def test_init_device_never_auto_selects_mps():
    device = init_device()
    assert device.type in ("cuda", "cpu"), (
        f"init_device() returned {device.type!r}; MPS must not be auto-selected "
        "because it cannot hold the float64 tensors DeepLens uses."
    )


def test_init_device_default_supports_float64():
    """float64 张量必须能够放置在自动选择的设备上。"""
    device = init_device()
    x = torch.zeros(2, dtype=torch.float64, device=device)  # 不得抛出异常
    assert x.dtype == torch.float64
