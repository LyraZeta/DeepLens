# 运行全部测试

在仓库根目录运行：

```bash
pytest test/ -v
python3 -m pytest test/ -v
```

# 运行指定测试文件

```bash
pytest test/test_ray.py -v
```

# 运行测试并生成覆盖率报告

```bash
pytest test/ --cov=deeplens --cov-report=term-missing
```
