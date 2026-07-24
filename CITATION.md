## 引用 DeepLens 论文

**可微光学**由 KAUST 计算成像研究组 (https://vccimaging.org/) 开发。首个可微光线追迹器版本由 [Congli Wang 博士](https://congliwang.github.io/)基于 [Mitsuba2](https://github.com/mitsuba-renderer/mitsuba2) 实现，相关论文如下：

```bibtex
@article{sun2021end,
  title={End-to-end complex lens design with differentiable ray tracing},
  author={Sun, Qilin and Wang, Congli and Qiang, Fu and Xiong, Dun and Wolfgang, Heidrich},
  journal={ACM Trans. Graph},
  volume={40},
  number={4},
  pages={1--13},
  year={2021}
}
```

随后，Congli Wang 博士实现了首个 PyTorch 版本的可微光线追迹器（[**dO**](https://github.com/vccimaging/DiffOptics)），相关论文如下：

```bibtex
@article{wang2022differentiable,
  title={do: A differentiable engine for deep lens design of computational imaging systems},
  author={Wang, Congli and Chen, Ni and Heidrich, Wolfgang},
  journal={IEEE Transactions on Computational Imaging},
  volume={8},
  pages={905--916},
  year={2022},
  publisher={IEEE}
}
```

目前，[Xinge Yang](https://singer-yang.github.io/) 正在开发和维护 [**DeepLens**](https://github.com/singer-yang/DeepLens/)。以下论文中的自动化镜头设计工作展示了可微光学相较于经典光学设计的突出能力：

```bibtex
@article{yang2024curriculum,
  title={Curriculum learning for ab initio deep learned refractive optics},
  author={Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang},
  journal={Nature communications},
  volume={15},
  number={1},
  pages={6572},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

以下论文提出了一种可微的**光线—波动模型**，用于仿真和优化折射—衍射混合镜头：

```bibtex
@inproceedings{yang2024end,
  title={End-to-end hybrid refractive-diffractive lens design with differentiable ray-wave model},
  author={Yang, Xinge and Souza, Matheus and Wang, Kunyi and Chakravarthula, Praneeth and Fu, Qiang and Heidrich, Wolfgang},
  booktitle={SIGGRAPH Asia 2024 Conference Papers},
  pages={1--11},
  year={2024}
}
```

以下论文开发了可微的**非序列**光线追迹和**偏振追迹**方法：

```bibtex
@article{yang2026waveguide,
  title={End-to-end differentiable design of geometric waveguide displays},
  author={Yang, Xinge and Liu, Zhaocheng and Nie, Zhaoyu and Fan, Qingyuan and Shi, Zhimin and Bonar, Jim and Heidrich, Wolfgang},
  journal={arXiv preprint arXiv:2601.04370},
  year={2026}
}
```
