# 科学计算工具

本目录提供分子构建、拓扑转换、模拟、轨迹分析、独立校验、审计和异常处理源码。

在仓库根目录安装科学依赖后，运行最小输入校验示例：

```bash
python science_core/examples/minimal_validation.py
```

预期输出为`{"valid": "pass", "invalid": "reject"}`。

MCP入口为`generated_code/md_analysis/mcp/pelcb_mcp_server.py`。通过`PELCB_PROJECT_CONFIG`指定自己的项目JSON，填写输入、输出、SSH账户和科学软件路径。新项目使用`generated_code/md_analysis/configs/pelcb100.public.json`作为配置模板。

远程模拟需要EMC、GROMACS 2024.4和可用计算资源；审计模型按所用入口单独配置。计算产物保存在任务目录。

- [连接Codex](../docs/CODEX_SETUP.md)
- [连接服务器](../docs/REMOTE_SETUP.md)
- [配置模型](../docs/MODEL_SETUP.md)
- [配置材料与分析](../docs/MATERIAL_SETUP.md)
