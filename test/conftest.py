"""
DeepLens 测试套件共享的 pytest fixture。
"""

import copy
import os
import sys

import pytest
import torch

# 将 deeplens 添加到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# 设备 fixture
# =============================================================================
@pytest.fixture(scope="session")
def device():
    """若 CUDA 可用则返回 CUDA 设备，否则返回 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        pytest.skip("CUDA not available, skipping GPU tests")


@pytest.fixture(scope="session")
def device_cpu():
    """返回 CPU 设备。"""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def device_auto():
    """若 CUDA 可用则返回 CUDA，否则返回 CPU（用于可在任一设备运行的测试）。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# 镜头 fixture
# =============================================================================
@pytest.fixture(scope="function")
def sample_singlet_lens(_sample_singlet_lens_cached, device_auto):
    """加载用于测试的简单单透镜。"""
    lens = copy.deepcopy(_sample_singlet_lens_cached)
    lens.to(device_auto)
    return lens


@pytest.fixture(scope="session")
def _sample_singlet_lens_cached():
    """会话级缓存的单透镜模板。"""
    from deeplens import GeoLens

    lens_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets/lenses/singlet/example1.json",
    )
    return GeoLens(filename=lens_path)


@pytest.fixture(scope="function")
def sample_cellphone_lens(_sample_cellphone_lens_cached, device_auto):
    """加载用于测试、包含非球面的手机镜头。"""
    lens = copy.deepcopy(_sample_cellphone_lens_cached)
    lens.to(device_auto)
    return lens


@pytest.fixture(scope="session")
def _sample_cellphone_lens_cached():
    """会话级缓存的手机镜头模板。"""
    from deeplens import GeoLens

    lens_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets/lenses/cellphone/cellphone68deg.json",
    )
    return GeoLens(filename=lens_path)


@pytest.fixture(scope="function")
def sample_camera_lens(_sample_camera_lens_cached, device_auto):
    """加载用于测试的相机镜头。"""
    lens = copy.deepcopy(_sample_camera_lens_cached)
    lens.to(device_auto)
    return lens


@pytest.fixture(scope="session")
def _sample_camera_lens_cached():
    """会话级缓存的相机镜头模板。"""
    from deeplens import GeoLens

    lens_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets/lenses/camera/ef50mm_f1.8.json",
    )
    return GeoLens(filename=lens_path)


# =============================================================================
# 图像 fixture
# =============================================================================
@pytest.fixture(scope="function")
def sample_image(device_auto):
    """创建简单的测试图像张量 [B, C, H, W]。"""
    # 创建用于测试的渐变图像
    H, W = 256, 256
    x = torch.linspace(0, 1, W, device=device_auto)
    y = torch.linspace(0, 1, H, device=device_auto)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    
    img = torch.stack([xx, yy, (xx + yy) / 2], dim=0)  # [3, H, W]
    img = img.unsqueeze(0)  # [1, 3, H, W]
    return img


@pytest.fixture(scope="function")
def sample_image_small(device_auto):
    """创建用于快速测试的小型图像张量。"""
    H, W = 64, 64
    img = torch.rand(1, 3, H, W, device=device_auto)
    return img


# =============================================================================
# 光线 fixture
# =============================================================================
@pytest.fixture(scope="function")
def sample_ray(device_auto):
    """创建用于测试的样例光线。"""
    from deeplens.light import Ray

    o = torch.tensor([[0.0, 0.0, -100.0]], device=device_auto)
    d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
    ray = Ray(o, d, wvln=0.55, device=device_auto)
    return ray


@pytest.fixture(scope="function")
def sample_rays_batch(device_auto):
    """创建一批用于测试的光线。"""
    from deeplens.light import Ray

    # 以网格模式创建 100 条光线
    n = 10
    x = torch.linspace(-1, 1, n, device=device_auto)
    y = torch.linspace(-1, 1, n, device=device_auto)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    
    o = torch.stack([xx.flatten(), yy.flatten(), torch.full((n*n,), -100.0, device=device_auto)], dim=-1)
    d = torch.zeros_like(o)
    d[..., 2] = 1.0
    
    ray = Ray(o, d, wvln=0.55, device=device_auto)
    return ray


# =============================================================================
# 路径辅助函数
# =============================================================================
@pytest.fixture(scope="session")
def project_root():
    """返回项目根目录。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def lenses_dir(project_root):
    """返回镜头数据集目录。"""
    return os.path.join(project_root, "datasets/lenses")


@pytest.fixture(scope="session")
def test_output_dir(project_root):
    """返回测试输出目录。"""
    output_dir = os.path.join(project_root, "test/test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =============================================================================
# HybridLens / DiffractiveLens 测试夹具
# =============================================================================
@pytest.fixture(scope="function")
def sample_hybridlens(_sample_hybridlens_cached, device_auto):
    """加载用于测试的混合镜头。"""
    lens = copy.deepcopy(_sample_hybridlens_cached)
    lens.to(device_auto)
    return lens


@pytest.fixture(scope="session")
def _sample_hybridlens_cached():
    """会话级缓存的混合镜头模板。"""
    from deeplens import HybridLens

    lens_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets/lenses/hybridlens/a489_doe.json",
    )
    old_dtype = torch.get_default_dtype()
    try:
        return HybridLens(filename=lens_path)
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.fixture(scope="function")
def sample_diffraclens():
    """创建用于测试的衍射镜头。"""
    from deeplens import DiffractiveLens
    from deeplens.diffractive_surface import Fresnel

    old_dtype = torch.get_default_dtype()
    lens = DiffractiveLens()
    lens.surfaces = [Fresnel(f0=50, d=0, res=500, fab_ps=0.008)]
    lens.d_sensor = torch.tensor(50.0, dtype=torch.float64)
    lens.foclen = float(lens.d_sensor)
    lens.sensor_size = (4.0, 4.0)
    lens.sensor_res = (500, 500)
    lens.pixel_size = lens.sensor_size[0] / lens.sensor_res[0]
    # 将表面移动到镜头所在设备（可能为 CUDA）
    lens.surfaces[0].to(lens.device)
    torch.set_default_dtype(old_dtype)
    return lens
