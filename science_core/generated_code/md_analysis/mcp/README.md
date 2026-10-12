# MCP 工具入口

`pelcb_mcp_server.py` 使用 stdio 提供智能体可调用的工具。源码中的 `@mcp.tool()` 函数定义实际工具名、参数与返回值。

在公开仓库根目录按 [安装与运行说明](../../../../README.md) 准备 Python 环境，并读取 [科学核心说明](../../../README.md)。复制 [公开配置](../configs/pelcb100.public.json) 为自己的本地配置，填写账户、主机、节点、输入与输出路径，以及 GROMACS 2024.4 的可执行文件位置。通过环境变量 `PELCB_PROJECT_CONFIG` 指向本地配置的绝对路径。

将以下命令登记到 MCP 客户端，作为 stdio 服务入口：

```bash
python science_core/generated_code/md_analysis/mcp/pelcb_mcp_server.py
```

客户端按实际工具签名传递参数。改变状态的工具要求 `execute=true`、已登记的 `project_id` 和 `allowlist_tag`，以及至少 12 个字符的具体 `reason`。规划模型选择允许的下一步，执行客户端调用工具并保存实际产物；科学状态由登记规则给出。

按以下顺序查看实现：

1. 设计与构建：`validate_polymer_design_candidate`、`validate_or_plan_polymer_input`、`materialize_polymer_input`。
2. 计划与执行：`plan_md_package`、`submit_supervisor_case`、`submit_md_clean_engine`、`run_postprocess_closure`。
3. 科学校验：`apply_polymer_qc_rules`、`get_rule_profile`、`apply_rule_gate`。
4. 审计与处理：`run_warn_audit_diagnosis`、`list_repair_tool_affordances`、`validate_repair_tool_call`、`plan_repair_route`。

配置决定项目、路径和执行范围。读取实际返回值及输出文件，区分计划、提交、运行完成与科学验收。构建器和规则资源位于上一级 `plugins/`。
