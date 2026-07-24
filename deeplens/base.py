"""所有可微光学对象的 DeepObj 基类。"""

import copy

import torch
import torch.nn as nn


class DeepObj:
    """DeepLens 中所有可微光学对象的基类。

    通过自动检查实例张量和嵌套的 `DeepObj` 子对象，提供设备管理、dtype 转换
    和深拷贝支持。所有镜头、表面、材料、光线和波对象均继承自此类。

    属性:
        dtype (torch.dtype): 所有持有张量的浮点 dtype。
        device (str or torch.device): 计算设备，由 `to` 设置。
    """

    def __init__(self, dtype=None):
        """初始化基础对象并记录其浮点 dtype。

        参数:
            dtype (torch.dtype, optional): 所持有张量的浮点 dtype。为 None 时
                默认为 `torch.get_default_dtype()`。
        """
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

    def __str__(self):
        """返回列出对象属性的多行字符串。

        标量和张量按 `key: value` 输出；列表和元组逐元素展开；字典和集合则跳过。

        返回:
            text (str): 便于阅读的对象属性摘要。
        """
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if val.__class__.__name__ in ["list", "tuple"]:
                for i, v in enumerate(val):
                    lines += "{}[{}]: {}".format(key, i, v).split("\n")
            elif val.__class__.__name__ in ["dict", "OrderedDict", "set"]:
                pass
            else:
                lines += "{}: {}".format(key, val).split("\n")

        return "\n    ".join(lines)

    def __call__(self, inp):
        """将输入转发给子类的 `forward` 方法。

        参数:
            inp (Any): 传递给 `self.forward` 的输入。

        返回:
            output (Any): `self.forward(inp)` 的结果。
        """
        return self.forward(inp)

    def clone(self):
        """返回此对象的深拷贝。

        返回:
            obj (DeepObj): `self` 的全新独立深拷贝。
        """
        return copy.deepcopy(self)

    def to(self, device):
        """将所有张量和嵌套对象移动到指定设备。

        递归遍历每个实例属性，将张量、`nn.Parameter` 数据、`nn.Module` 子对象、
        嵌套的 `DeepObj` 对象以及列表和元组中的张量/`DeepObj` 项移动到目标设备。

        参数:
            device (str or torch.device): 目标设备，例如 `"cuda"`、`"cpu"`
                或 `torch.device` 实例。

        返回:
            self (DeepObj): 更新后的对象（便于链式调用）。

        示例:
            ```python
            lens = GeoLens(filename="lens.json")
            lens.to("cuda")  # 将所有张量移动到 GPU
            ```
        """
        self.device = device

        for key, val in vars(self).items():
            if isinstance(val, nn.Parameter):
                val.data = val.data.to(device)
            elif torch.is_tensor(val):
                setattr(self, key, val.to(device))
            elif isinstance(val, nn.Module):
                val.to(device)
            elif issubclass(type(val), DeepObj):
                val.to(device)
            elif val.__class__.__name__ in ("list", "tuple"):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        val[i] = v.to(device)
                    elif issubclass(type(v), DeepObj):
                        v.to(device)
        return self

    def astype(self, dtype):
        """将所有浮点张量转换为目标 dtype。

        递归转换所持有的浮点张量、`nn.Parameter` 数据和嵌套的 `DeepObj` 对象
        （包括列表中的对象）。当 dtype 与当前默认值不同时，还会调用
        `torch.set_default_dtype(dtype)`，使后续创建的张量与之匹配。

        参数:
            dtype (torch.dtype or None): 目标浮点 dtype，可为 `torch.float16`、
                `torch.float32` 或 `torch.float64`。为 None 时不执行操作，
                原样返回 `self`。

        返回:
            self (DeepObj): 更新后的对象（便于链式调用）。

        异常:
            AssertionError: dtype 不是上述三种受支持的浮点 dtype 之一。

        示例:
            ```python
            lens = GeoLens(filename="lens.json")
            lens.astype(torch.float64)  # 切换为双精度
            ```
        """
        if dtype is None:
            return self

        dtype_ls = [torch.float16, torch.float32, torch.float64]
        assert dtype in dtype_ls, f"Data type {dtype} is not supported."

        if torch.get_default_dtype() != dtype:
            torch.set_default_dtype(dtype)
            print(f"Set {dtype} as default torch dtype.")

        self.dtype = dtype
        for key, val in vars(self).items():
            if isinstance(val, nn.Parameter):
                if val.dtype in dtype_ls:
                    val.data = val.data.to(dtype)
            elif torch.is_tensor(val) and val.dtype in dtype_ls:
                setattr(self, key, val.to(dtype))
            elif issubclass(type(val), DeepObj):
                val.astype(dtype)
            elif issubclass(type(val), list):
                for i, v in enumerate(val):
                    if torch.is_tensor(v) and v.dtype in dtype_ls:
                        val[i] = v.to(dtype)
                    elif issubclass(type(v), DeepObj):
                        v.astype(dtype)
        return self
