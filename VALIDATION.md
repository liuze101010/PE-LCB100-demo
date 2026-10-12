# 安装检查

使用Python 3.12安装科学工具依赖后，在源码目录运行：

```bash
python science_core/examples/minimal_validation.py
python -m polymer_agent_demo.scripts.doctor --stage local-validation --json
```

输入校验示例应返回`{"valid": "pass", "invalid": "reject"}`。环境检查会列出依赖和配置情况。

远程计算前，按[服务器说明](docs/REMOTE_SETUP.md)检查SSH、EMC、GROMACS、工作目录和可用资源。Codex接入见[连接说明](docs/CODEX_SETUP.md)。
