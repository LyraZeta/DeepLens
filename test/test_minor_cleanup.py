"""次要问题清理的回归测试（低严重性批次）。

清理前的代码会使每项测试失败（抛出异常/产生 NaN/类型错误）。
"""

import json

import torch

from deeplens.geometric_surface import Mirror
from deeplens.material import Material
from deeplens.phase_surface import Binary2Phase
from deeplens.light import Ray
from deeplens import utils


def test_gpu_init_uses_supported_api():
    """gpu_init 过去使用近期 torch 已移除的 torch.set_default_tensor_type。"""
    old = torch.get_default_dtype()
    try:
        device = utils.gpu_init()
        assert isinstance(device, torch.device)
    finally:
        torch.set_default_dtype(old)


def test_mirror_surf_dict_is_json_serializable():
    """Mirror.surf_dict 过去为 'd' 存储 torch.Tensor -> 无法进行 JSON 序列化。"""
    surf = Mirror(r=5.0, d=10.0)
    sd = surf.surf_dict()
    assert isinstance(sd["d"], float)
    json.dumps(sd)  # 不得抛出异常


def test_set_sellmeier_param_switches_dispersion():
    """set_sellmeier_param 过去未设置 self.dispersion，因此 ior() 会无提示地忽略
    这些系数。"""
    mat = Material("1.5/50.0")  # 构造为 Cauchy 材料
    assert mat.dispersion == "cauchy"
    mat.set_sellmeier_param()
    assert mat.dispersion == "sellmeier"


def test_phase_plane_intersect_parallel_ray_grad_is_finite():
    """Phase.intersect 过去会在无保护的情况下除以 d_z。无效（平行）光线在前向
    传播中通过 torch.where 保持原点，但无保护的 t = .../d_z
    （d_z == 0 -> inf）会使反向传播产生 NaN 梯度（掩码 where 分支中的 0 * inf）。"""
    o = torch.tensor([[0.5, 0.0, -1.0]], requires_grad=True)
    d = torch.tensor([[1.0, 0.0, 0.0]])  # 平行于平面：d_z == 0
    surf = Binary2Phase(r=2.0, d=0.0)
    ray = Ray(o, d, wvln=0.55)
    ray = surf.intersect(ray)
    ray.o.sum().backward()
    assert torch.isfinite(o.grad).all()
