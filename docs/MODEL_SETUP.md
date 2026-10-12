# 模型连接

Codex负责理解任务并调用科学工具，审计模型负责分析校验发现的问题。Codex登录与MCP连接见[连接说明](CODEX_SETUP.md)。

## MCP审计模型

向MCP进程提供以下环境变量：

| 模型 | 密钥字段 | 型号字段 | 默认型号 |
| --- | --- | --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_MODEL` | `deepseek-v4-pro` |
| Qwen | `DASHSCOPE_API_KEY` | `QWEN_MODEL` | `qwen3.7-max` |
| GLM | `ZHIPUAI_API_KEY` | `GLM_MODEL` | `glm-4.6` |

型号填写自己账户可以使用的型号。GLM还兼容`ZHIPU_API_KEY`、`GLM_API_KEY`及型号字段`ZHIPU_MODEL`。

客户端向MCP传入环境变量；使用个人`.env`文件时，由启动方式载入。密钥保存在个人环境中。Codex的登录凭据与这些审计API分别配置。

`run_warn_audit_diagnosis`和`run_warn_audit_batch`提供审计入口。`execute=false`查看本地诊断；实际调用使用`execute=true`和`use_mock_clients=false`，并填写工具要求的项目标识、标签及操作原因。记录保存实际型号与返回结果。

## 可选网页工作台

| 用途 | 配置字段 |
| --- | --- |
| 规划模型密钥 | `OPENAI_API_KEY`，或个人账号在页面保存的密钥 |
| 启用自然语言设计 | `POLYMER_AGENT_LIVE_DESIGN=1` |
| 设计型号 | `POLYMER_AGENT_DESIGN_MODEL`，默认`gpt-5.5` |
| 执行与修复规划型号 | `POLYMER_AGENT_UPPER_PLANNER_MODEL`，默认使用设计型号 |
| DeepSeek审计型号 | `POLYMER_AGENT_AUDIT_DEEPSEEK_MODEL` |
| Qwen审计型号 | `POLYMER_AGENT_AUDIT_QWEN_MODEL` |
| GLM审计型号 | `POLYMER_AGENT_AUDIT_GLM_MODEL` |

工作台的审计密钥使用`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`和`ZHIPUAI_API_KEY`，也可以在个人账号页面保存。密钥加密存储，页面显示配置状态。

工作台启动时可以从环境填入管理员的空白密钥项；设置`POLYMER_AGENT_SEED_KEYS_FROM_ENV=0`关闭此方式。直接使用MCP时，型号采用`DEEPSEEK_MODEL`等字段；网页工作台采用`POLYMER_AGENT_AUDIT_*_MODEL`字段。
