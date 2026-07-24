# 贡献指南

感谢您有兴趣为 DeepLens 作出贡献！

所有贡献者都应遵守我们的[行为准则](./CODE_OF_CONDUCT.md)。

## 贡献者许可协议（CLA）

向 DeepLens 项目提交拉取请求的所有贡献者都必须签署贡献者许可协议（CLA）。此流程通过 [CLA Assistant](https://cla-assistant.io/) 自动完成；当您首次提交拉取请求时，它会提示您签署 CLA。您可以查看 [DeepLens-CLA](https://gist.github.com/singer-yang/b2e4214a12a220899ed682d9c24f575b)。

## 如何贡献

我们欢迎各种形式的贡献，包括但不限于：

- 报告错误
- 提交包含错误修复或新功能的拉取请求
- 改进文档
- 添加新示例或教程

如果您计划开发一项重要功能，请先创建议题，与维护者讨论您的想法。

## 开发环境安装

DeepLens 主要是一个 PyTorch 项目。要配置开发环境，请按照 [README.md](./README.md) 中的“安装”章节创建 conda 环境并安装必要的依赖项。

步骤概要如下：
```
# 创建并激活 conda 环境
conda env create -f environment.yml -n deeplens_env
conda activate deeplens_env
```
或者
```
conda create --name deeplens_env python=3.9
conda activate deeplens_env
pip install -r requirements.txt
```

## 代码格式化

我们鼓励贡献者使用 [ruff](https://docs.astral.sh/ruff/) 格式化代码，以便在整个项目中保持一致的代码风格。您可以通过以下命令安装 ruff 并格式化代码：

```
pip install ruff
ruff format .
```

## 贡献机会

项目在 GitHub 上的议题跟踪器是寻找贡献思路的良好起点。您也可以查看 README 中提到的[开放问题项目面板](https://github.com/users/singer-yang/projects/2)。

## 提议重大变更

如果要对代码库进行重大更改，建议先创建议题并提出变更方案。这样，您可以在投入大量时间实现之前与维护者和社区进行讨论，从而确保您的贡献符合项目目标，并提高其被接受的可能性。
