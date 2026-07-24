"""效率重构的特征测试（根因 G）。

这些测试固定优化必须保留的*行为*：使用缓存的旋转矩阵必须与重新构建矩阵等价，
局部/全局变换也必须仍可往返转换。
"""

import torch

from deeplens.geometric_surface.plane import Plane
from deeplens.light import Ray
from deeplens.phase_surface import NURBSPhase


def test_cached_rotation_matrices_equal_freshly_built():
    """to_local/to_global 现在使用缓存的 _R_*；缓存必须等于旧代码每次调用时
    重新构建的矩阵。"""
    surf = Plane(r=5.0, d=10.0, mat2="air", vec_local=[0.1, -0.2, 1.0])
    assert surf._R_to_local is not None  # 倾斜表面需要旋转
    assert torch.allclose(
        surf._R_to_local, surf._get_rotation_matrix(surf.vec_local, surf.vec_global)
    )
    assert torch.allclose(
        surf._R_to_global, surf._get_rotation_matrix(surf.vec_global, surf.vec_local)
    )


def test_local_global_roundtrip_on_tilted_surface():
    """to_global_coord(to_local_coord(ray)) 应返回原始光线（缓存矩阵互为精确逆矩阵）。"""
    surf = Plane(r=5.0, d=10.0, mat2="air", vec_local=[0.1, -0.2, 1.0])
    o = torch.tensor([[0.3, -0.4, -2.0], [1.0, 0.5, 3.0]])
    d = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 0.99]])
    ray = Ray(o.clone(), d.clone(), wvln=0.55)

    o0, d0 = ray.o.clone(), ray.d.clone()
    ray = surf.to_global_coord(surf.to_local_coord(ray))

    assert torch.allclose(ray.o, o0, atol=1e-5)
    assert torch.allclose(ray.d, d0, atol=1e-5)


def test_on_axis_surface_has_no_rotation_cache():
    """轴上表面无需旋转；缓存为 None（无操作路径）。"""
    surf = Plane(r=5.0, d=10.0, mat2="air")  # vec_local 默认为 [0,0,1]
    assert surf._R_to_local is None
    assert surf._R_to_global is None


def test_set_fnum_still_hits_target_with_reduced_pupil_sampling(sample_cellphone_lens):
    """光瞳光线扇从 1024 条减少到 SPP_PUPIL=128 条（O(N^2) 估计器）。
    set_fnum 仍必须收敛到请求的 F-number。"""
    lens = sample_cellphone_lens
    lens.set_fnum(4.0)
    _, pupil_r = lens.calc_entrance_pupil()
    achieved = lens.foclen / (2.0 * pupil_r)
    assert abs(achieved - 4.0) / 4.0 < 0.01  # 与目标值的偏差在 1% 以内


def _nurbs_phi_per_point(surf, x, y):
    """参考实现：原始逐点 NURBS phi（循环调用 _evaluate_nurbs_surface）。"""
    x_norm = (x / surf.norm_radii + 1.0) / 2.0
    y_norm = (y / surf.norm_radii + 1.0) / 2.0
    xf, yf = x_norm.flatten(), y_norm.flatten()
    z = torch.stack(
        [surf._evaluate_nurbs_surface(xf[i], yf[i])[2] for i in range(xf.numel())]
    ).reshape(x_norm.shape)
    r2 = (x / surf.norm_radii) ** 2 + (y / surf.norm_radii) ** 2
    z = torch.where(r2 > 1, torch.zeros_like(z), z)
    return torch.remainder(z, 2 * torch.pi)


def test_nurbs_phi_vectorized_matches_per_point():
    """向量化 phi() 必须逐元素等于逐点循环结果。"""
    torch.manual_seed(0)
    ncp = 6
    cp = torch.randn(ncp, ncp, 3) * 0.3  # phi 仅取决于 z（相位）
    weights = torch.rand(ncp, ncp) + 0.5  # 正权重
    surf = NURBSPhase(
        r=2.0, d=0.0,
        control_points_u=ncp, control_points_v=ncp,
        degree_u=3, degree_v=3,
        control_points=cp, weights=weights,
    )

    xs = torch.linspace(-1.8, 1.8, 7)
    ys = torch.linspace(-1.8, 1.8, 5)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")

    out = surf.phi(X, Y)                     # 向量化结果
    ref = _nurbs_phi_per_point(surf, X, Y)   # 逐点参考结果
    assert out.shape == X.shape
    assert torch.allclose(out, ref, atol=1e-4)
