<div align="center">
    <img src="assets/logo.png" alt="DeepLens logo" width="400px" >
</div>

# DeepLens

[简体中文](./README.md)

DeepLens is a differentiable optical lens simulator for end-to-end computational imaging, supporting multiple optical models (eg., geometric ray tracing, diffractive wave propagation, hybrid ray-wave model, surrogate PSF network).

DeepLens can be used for (1) end-to-end optics-algorithm co-design, (2) gradient-based automated optical design, and (3) synthetic dataset generation via image simulation. DeepLens enables researchers to rapidly prototype and optimize custom optical systems.

<p align="center">
    <a href="https://vccimaging.org/DeepLens/"><img src="https://img.shields.io/badge/Docs-blue?style=flat&logo=readthedocs&logoColor=white" alt="Docs"/></a>
    <a href="https://github.com/singer-yang/DeepLens-tutorials"><img src="https://img.shields.io/badge/Tutorials-black?style=flat&logo=github&logoColor=white" alt="Tutorials"/></a>
    <a href="#community"><img src="https://img.shields.io/badge/Community-Slack-4A154B?style=flat&logo=slack&logoColor=white" alt="Community"/></a>
    <a href="https://pypi.org/project/deeplens-core/"><img src="https://img.shields.io/pypi/v/deeplens-core?label=PyPI&color=orange&logo=pypi&logoColor=white" alt="PyPI"/></a>
    <a href="https://deepwiki.com/singer-yang/DeepLens"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"/></a>
</p>

## Features

1. **Differentiable Optics.** DeepLens leverages differentiable optical simulation to enable accurate, efficient gradient calculation for lens inverse design.
2. **Automated Design.** DeepLens enables fully automated optical design via gradient-based and advanced optimization algorithms, shortening the development cycle for a wide range of optical systems (e.g., highly aspherical lenses, metasurfaces, and AR/VR displays).
3. **Multiple Optical Models.** DeepLens supports geometric ray tracing alongside hybrid ray-wave models, neural lens representations, and interpolation-based models.
4. **Image Simulation.** DeepLens delivers photorealistic image rendering with spatially varying, depth-dependent aberrations, closing the sim-to-real gap when combined with [End2end-Imaging](https://github.com/vccimaging/End2endImaging).

Additional features (customizable upon request):

1. **GPU Kernel Acceleration.** Achieves >10x speedup and >90% GPU memory reduction with custom GPU kernels across NVIDIA and AMD platforms, making deployment on local laptops practical.
2. **Polarization Ray Tracing.** Supports polarization ray tracing and inverse design of thin films via [DiffTMM](https://github.com/AI4Optics/DiffTMM).
3. **Non-Sequential Ray Tracing.** Supports a differentiable non-sequential ray tracing model for stray light analysis and optimization.
4. **Distributed Optimization.** Supports distributed simulation and optimization for billion-scale ray tracing and high-resolution (>100k x 100k) diffractive propagation.

## Applications

#### 1. Lens Analysis and Image Simulation

DeepLens supports comprehensive lens analysis (spot diagram, PSF, MTF, distortion, etc.) and photorealistic image simulation with spatially-varying, depth-dependent aberrations.

<div align="center">
    <img src="assets/feature.png" alt="Lens Analysis and Image Simulation"/>
</div>

#### 2. Automated geometric lens design

Fully automated lens design from scratch with gradient-based optimization and advanced optimization algorithms.

> **Note:** Automated lens design is now actively maintained in the [**AutoLens**](https://github.com/AI4Optics/AutoLens) project. If your focus is automated lens design, we recommend using the AutoLens repo instead, as it receives dedicated updates and improvements for this use case.

[![paper](https://img.shields.io/badge/NatComm-2024-orange)](https://www.nature.com/articles/s41467-024-50835-7) [![quickstart](https://img.shields.io/badge/AutoLens-green)](https://github.com/AI4Optics/AutoLens)

<div align="center">
    <img src="assets/autolens1.gif" alt="AutoLens" height="270px"/>
    <img src="assets/autolens2.gif" alt="AutoLens" height="270px"/>
</div>

#### 3. Neural Lens PSF Representation

A surrogate network for efficient lens PSF representation, supporting fast and accurate image simulation with spatially-varying aberrations and defocus.

[![paper](https://img.shields.io/badge/TPAMI-2023-orange)](https://ieeexplore.ieee.org/document/10209238) [![link](https://img.shields.io/badge/Project-green)](https://github.com/vccimaging/Aberration-Aware-Depth-from-Focus)

<div align="center">
    <img src="assets/implicit_net.png" alt="Neural lens PSF representation" height="150px"/>
</div>

#### 4. Hybrid Ray-Wave Optical Model

Differentiable ray-wave optical model for accurate lens aberration and diffraction element simulation, supporting end-to-end refractive-diffractive lens design.

[![paper](https://img.shields.io/badge/SiggraphAsia-2024-orange)](https://dl.acm.org/doi/10.1145/3680528.3687640)

<div align="center">
    <img src="assets/hybridlens.png" alt="Hybrid ray-wave optical model" height="200px"/>
</div>

#### 5. Non-sequential Model and Polarization Tracing

Non-sequential polarization tracing to accurately simulate the polarization state of light passing through a geometric waveguide AR display. End-to-end optimization for coating film inverse design targeting the out-coupling eyebox response.

<div align="center">
    <img src="assets/diffgwg.jpg" alt="Non-sequential polarization ray tracing for AR waveguide display" height="200px"/>
</div>

#### 6. End-to-End Computational Imaging

DeepLens serves as the differentiable optics engine in [**End2endImaging**](https://github.com/vccimaging/End2endImaging), an end-to-end differentiable computational imaging framework. End2endImaging integrates optics, sensor/ISP simulation, and neural reconstruction networks into a single PyTorch computation graph, enabling joint optimization of the entire camera pipeline.

<div align="center">
    <img src="assets/end2end.png" alt="End2endImaging" height="200px"/>
</div>

## Installation

Clone this repo:

```
git clone https://github.com/singer-yang/DeepLens
cd DeepLens
```

Create a conda environment:

```
conda create -n deeplens_env python=3.12
conda activate deeplens_env

# Linux and Mac
pip install torch torchvision
# Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

or

```
conda env create -f environment.yml -n deeplens_env
```

Run the demo code:

```
python 0_hello_geolens.py
```

## MWIR telescope design entry point

The repository includes a first-order specification checker and a GeoLens initializer for
transmissive 2.7–4.3 µm systems:

```powershell
conda activate deeplens_env
python mwir_spec.py
python mwir_spec.py --json
python mwir_telescope_design.py --check-only
python mwir_telescope_design.py --device cpu --iterations 0 --output results\mwir-initial
```

The default configuration no longer enables the 42 µrad two-pixel constraint. It follows the
Zemax summary: a Y-direction full field from -4.8° to +4.8° (9.6° total), a 47.1454 mm
half-image-height, and a 280 mm entrance pupil. The focal length is derived as approximately
561.44 mm (F/2.005). The detector format is intentionally left unspecified; the default
`transmission_baseline` uses a circular-equivalent virtual sensor only for initial numerical
design, not as a final detector requirement. It is approximately 66.67 × 66.67 mm, with a
94.2908 mm diagonal corresponding to the full Y-image-height envelope.

Use the design entry point to check overridden field, image-height, and pupil values:

```powershell
python mwir_telescope_design.py --check-only `
  --field-y-deg 9.6 `
  --image-height-mm 47.1454 `
  --entrance-pupil-mm 280
```

### Initial prescription, optimization, and evaluation

Generate only the initial prescription:

```powershell
python mwir_telescope_design.py --device cpu --iterations 0 `
  --output results\mwir-initial
```

The initializer calibrates the combined paraxial power from the measured focal length and then
refocuses at infinity. Gradient optimization uses a 100 km finite conjugate as an infinity
approximation; formal MTF and image-height/distortion evaluation trace parallel rays at infinity.
A low-sampling numerical check is available:

```powershell
python mwir_telescope_design.py --device cpu --iterations 0 `
  --evaluate --eval-spp 64 `
  --output results\mwir-initial-eval
```

A minimal CPU optimization smoke test should explicitly reduce sampling. `--iterations N` now
performs exactly N parameter updates:

```powershell
python mwir_telescope_design.py --device cpu --iterations 1 `
  --num-ring 2 --num-arm 2 --spp 32 `
  --evaluate --eval-spp 64 `
  --output results\mwir-opt-smoke
```

Use `--input-lens` to start a new stage from an existing JSON prescription. This restores only
the optical prescription: it does not recalibrate power, refocus, or restore the previous Adam
state. Wavelengths, object distance, front stop, 280 mm entrance pupil, sensor radius,
resolution, and lens count are validated again, and MWIR mechanical constraints are reapplied.
The new output directory must be empty and separate from the source stage so old metadata,
checkpoints, and final prescriptions cannot be overwritten. The companion
`mwir_design_metadata.json` is also checked; changing the original field, image height, or target
focal length is rejected unless `--allow-retarget` is explicitly supplied. A practical curriculum
first stabilizes field mapping, then improves spot RMS:

```powershell
# Stage 1: freeze curvature and prioritize focal-length/image-height stability
python mwir_telescope_design.py `
  --input-lens results\mwir-initial\mwir_initial.json `
  --device cpu --iterations 20 `
  --lrs 2e-3 0 2e-4 2e-6 `
  --rms-weight 0.3 `
  --field-weight 1.5 --field-max-weight 2 `
  --num-ring 8 --num-arm 4 --spp 128 `
  --output results\mwir-stage-field --evaluate

# Stage 2: continue from the previous final prescription and improve image quality
python mwir_telescope_design.py `
  --input-lens results\mwir-stage-field\mwir_final.json `
  --device cpu --iterations 100 `
  --lrs 1e-3 1e-7 5e-4 2e-6 `
  --rms-weight 1 `
  --field-weight 1.5 --field-max-weight 2 `
  --num-ring 8 --num-arm 4 --spp 512 `
  --output results\mwir-stage-rms --evaluate
```

The first-stage default learning rates are `[2e-3, 2e-7, 2e-4, 2e-6]` for spacing,
curvature, conic constant, and aspheric coefficients. On CPU, first check a 1–5-step trend with
small `spp`; change to `--device cuda` only when CUDA is available in the active environment.

MWIR checkpoints save only `optimization/iter*.json` by default. Add
`--checkpoint-analysis` only when full checkpoint plots are needed. Surface-shape correction and
aperture pruning remain disabled during the first stage; enable `--shape-control` after the
prescription stabilizes, and use `--prune-surfaces` last. The target image-height/field-mapping
loss is independent of generic regularization and traces a differentiable chief ray through the
center of the front stop at all three training wavelengths, at infinity, in two meridional planes,
and on nine equally spaced field angles by default. This aligns training with the formal chief-ray
distortion check. Its default weight is 1.0; tune it with `--field-weight`.
`--field-max-weight` additionally emphasizes the worst field, while `--regularization-weight`
controls mechanical/profile regularization.

The current gradient objective includes spot RMS, valid-ray ratio, target field mapping, and
prescription regularization. MTF 0.3 is an `--evaluate` acceptance threshold, not a directly
differentiated MTF loss. The optimizer sanitizes local NaN/Inf gradients, clips each parameter
group independently, and rolls back non-finite or valid-ray-degrading updates. If a high-order
aspheric basis exceeds the float32 dynamic range on a large aperture, its coefficient is retained
but frozen; for this system, a18 is excluded from Adam while a4–a16 remain trainable.

### Numerical evaluation definition

`--evaluate` samples 2.7, 3.5, and 4.3 µm at Y fields 0°, 3.36°, and 4.8°. The early-stage
system-MTF estimate is:

```text
geometric ray-intercept OTF × ideal unobscured circular-aperture diffraction MTF ×
100%-fill rectangular-pixel MTF
```

This is not a rigorous wave-aberration/Huygens MTF, so a validated physical-optics model is still
required for final acceptance. Overall acceptance requires EFL and F/# errors no greater than
1%, target field-mapping error and conventional distortion no greater than 0.5%, system MTF at
least 0.3, minimum valid-ray ratio at least 0.7, entrance-pupil error no greater than 1%, and no
more than seven lenses. The current vignetting metric is only a valid-ray ratio; it excludes
cos⁴ falloff, material absorption, and coating losses.

First-order evaluation deliberately separates three quantities. Strict EFL is obtained by first
extrapolating the Gaussian paraxial focal plane from small-pupil-height axial rays, then
extrapolating symmetric small-field chief-ray plate scale at that plane. F/# and diffraction MTF
use this strict EFL. Conventional distortion uses a wavelength- and plane-specific local chief-ray
plate scale at the current sensor plane. Target field mapping always uses the fixed task focal
length of 561.4396 mm. `lens.foclen/lens.fnum` remain in `mwir_metrics.json` as cached DeepLens
diagnostics, but sensor defocus is no longer misreported as conventional distortion and sensor
plate scale cannot hide a true EFL drift. Acceptance uses the worse of the two meridional planes.
The current prescription contains only centered rotationally symmetric surfaces, so the sampled
0-to-+4.8° half field represents the negative half as well. Any future decenter or tilt support
must extend acceptance to both field signs.

Output files are conditional on the requested stage:

- `mwir_design_metadata.json` and `mwir_initial.json` are always written.
- `--iterations > 0` also writes `mwir_final.json` and `optimization/iter*.json`; checkpoint
  plots are produced only with `--checkpoint-analysis`.
- `--evaluate` writes `mwir_metrics.json`.
- `--analyze` produces a full analysis of the initial prescription before optimization.

### Detector and historical scenarios

Until the detector pitch is confirmed, the Nyquist frequency and system MTF in
`mwir_metrics.json` are marked as provisional virtual-sensor values; the array format may be
confirmed independently later. The 47.1454 mm value is the Y-direction half image height, not a
detector half-diagonal. The full active detector
height must therefore be 94.2908 mm; once an aspect ratio is known, its width is derived from
that height. Horizontal field, diagonal field, and the final detector model remain unspecified.

To reproduce the former 42 µrad scenario, enable it explicitly; this path is historical only:

```powershell
python mwir_telescope_design.py --scheme large_fpa `
  --two-pixel-resolution-urad 42 `
  --simulation-pixel-pitch-um 30 `
  --device cpu --iterations 0 `
  --output results\mwir-history-42urad
```

`cassegrain_equivalent` is currently only a transmissive-baseline alias that inherits the
Cassegrain first-order requirements. It does not import mirror curvatures, separations, central
obscuration, or mechanical length, and it is not an automatic reflective-to-transmissive
prescription converter. The recorded 20°C condition does not yet model thermo-optic coefficients,
thermal expansion, material absorption, coatings, or tolerances.

DeepLens repo structure:

```
DeepLens/
│
├── deeplens/
│   ├── lens.py             (base lens class)
│   ├── geolens.py          (refractive lens)
│   ├── hybridlens.py       (refractive + diffractive hybrid lens)
│   ├── diffraclens.py      (diffractive lens)
│   ├── defocuslens.py      (circle-of-confusion model)
│   ├── psfnetlens.py       (surrogate lens PSF model)
│   ├── ...
│   ├── geometric_surface/  (refractive and reflective surfaces)
│   ├── diffractive_surface/(diffractive surfaces)
│   ├── phase_surface/      (phase surfaces)
│   ├── light/              (Ray, ComplexWave)
│   ├── material/           (glass/plastic catalogs + refractiveindex.info data)
│   ├── imgsim/             (PSF convolution, monte carlo image simulation)
│   ├── geolens_pkg/        (eval, optim, vis, io mixins)
│   └── surrogate/          (MLP, Siren neural surrogates)
│
├── 0_hello_geolens.py     (introductory tutorial)
├── mwir_spec.py            (MWIR first-order specification checker)
├── mwir_telescope_design.py (MWIR initializer and optimization entry point)
├── ...
└── 9_diffractive_surfaces.py (diffractive-surface examples)
```

## Community

Join our [Slack](https://join.slack.com/t/deeplens/shared_invite/zt-2wz3x2n3b-plRqN26eDhO2IY4r_gmjOw) workspace and WeChat Group (singeryang1999) to connect with our core contributors, receive the latest industry updates, and be part of our community. For any inquiries, contact Xinge Yang (xinge.yang@kaust.edu.sa).

## Contribution

We welcome all contributions. To get started, please read our [Contributing Guide](./CONTRIBUTING.md) or check out [open questions](https://github.com/users/singer-yang/projects/2). All project participants are expected to adhere to our [Code of Conduct](./CODE_OF_CONDUCT.md). A list of contributors can be viewed in [Contributors](./CONTRIBUTORS.md) and below:

<a href="https://github.com/singer-yang/DeepLens/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=singer-yang/DeepLens" />
</a>

## Citation

If you use DeepLens in your research, please cite the paper. See more in [History of DeepLens](./CITATION.md).

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
