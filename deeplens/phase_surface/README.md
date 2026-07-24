# 相位面

相位面是一类由平面衬底和衍射图案构成的衍射面。

在 Zemax 等商业软件中，衍射面通常通过在标准折射光线上附加一个光线偏折角进行仿真。DeepLens 中的相位面也依据同一原理使用几何光学进行仿真。（DeepLens 还支持通过波动光学仿真的衍射面，请参阅 `deeplens/diffractive_surface/` 目录。这两个模块均表示衍射面，主要区别在于仿真方法。）衍射图案也可以应用于曲面，但此功能尚未实现。

相位面的常见制造方法包括：
- **光刻**：标准半导体加工技术。
    - **蚀刻**：从衬底去除材料以形成衍射图案的减材工艺，通常需要多个步骤才能构成多级结构。
    - **灰度光刻**：使用光密度渐变的掩模，通过一次曝光和蚀刻形成连续或多级轮廓的技术。
- **纳米压印光刻（NIL）**：经济高效的复制方法。
- **单点金刚石车削（SPDT）**：仅适用于长波长（例如 >10µm）的 DOE 制造。

本模块的核心是 `phase.py` 中的 `Phase` 基类，它定义了所有相位面的通用接口，并处理光线追迹逻辑、坐标变换和衍射仿真。

## 可用表面

当前子模块中定义了以下表面，它们均继承自 `Phase` 基类。除 `QuarticPhase` 外，其余类均由 `phase_surface/__init__.py` 导出；`QuarticPhase` 目前需从 `deeplens.phase_surface.qphase` 导入。

-   `Phase`：所有相位面的基类。
    -   `Binary2Phase`：使用偶次多项式（$r^2, r^4, \dots$）表示旋转对称相位轮廓。
    -   `CubicPhase`：使用三次多项式（$x^3, y^3, x^2y, \dots$）实现三次相位轮廓。
    -   `FresnelPhase`：模拟由焦距定义的菲涅耳镜头相位轮廓。
    -   `GratingPhase`：表示由斜率和方向角定义的线性衍射光栅。
    -   `NURBSPhase`：使用非均匀有理 B 样条（NURBS）定义自由曲面相位轮廓。
    -   `PolyPhase`：通用多项式相位面，同时包含偶次径向项（如 `Binary2Phase`）和奇次多项式项。
    -   `QuarticPhase`：使用四次多项式系数实现 Q 型相位面。
    -   `VortexPhase`：使用拓扑荷定义涡旋相位轮廓。
    -   `ZernikePhase`：使用 Zernike 多项式表示相位轮廓（最多支持 37 项）。

常见的实际应用包括衍射光学元件（DOEs）和超表面。佳能的 DO（Diffractive Optics，衍射光学）镜头（https://www.canon-europe.com/pro/infobank/lenses-multi-layer-diffractive-optical-element/）便是一个广为人知的应用实例。
