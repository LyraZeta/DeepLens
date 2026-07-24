from deeplens import GeoLens
from deeplens.geometric_surface import Prism

# 一个薄透镜
lens = GeoLens(filename="./thinlens.json")

# 向镜头添加棱镜
prism = Prism(r=7.5, d=20.0, mirror_angle=45.0, mat2="bk7", device=lens.device)
lens.surfaces.append(prism)

# 光线追迹（经过 thinlens 后，入射棱镜的光线应平行于 +z）
ray = lens.sample_from_points(points=[[0.0, 0.0, -10.0]], num_rays=1024)
ray, _ = lens.trace(ray)

# 光线方向应朝上
print(ray.d)
