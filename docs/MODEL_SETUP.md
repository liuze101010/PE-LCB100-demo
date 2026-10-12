# 模型连接

规划模型和三模型审计承担不同步骤。Codex的登录、模型选择及MCP登记见[Codex连接说明](CODEX_SETUP.md)；服务器与项目路径见[服务器连接说明](REMOTE_SETUP.md)。

## 直接连接原MCP

本地结构校验和工具列表读取可以先完成。调用真实三模型审计时，向MCP进程提供以下环境变量：

| 模型 | 密钥字段 | 型号字段 | 保留的默认型号 |
| --- | --- | --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_MODEL` | `deepseek-v4-pro` |
| Qwen | `DASHSCOPE_API_KEY` | `QWEN_MODEL` | `qwen3.7-max` |
| GLM | `ZHIPUAI_API_KEY` | `GLM_MODEL` | `glm-4.6` |

原GLM客户端也接受`ZHIPU_API_KEY`或`GLM_API_KEY`，型号可用`ZHIPU_MODEL`作为兼容字段。建议新配置统一使用表中字段。

原MCP服务从进程环境读取这些值。个人`.env`文件须由自己使用的启动方式或客户端载入。密钥保存在个人配置中，版本库保存字段名称和空白示例。

真实审计入口为`run_warn_audit_diagnosis`或`run_warn_audit_batch`。设置`execute=false`可以查看本地规则诊断；真实调用采用`execute=true`、`use_mock_clients=false`及工具要求的项目标识、标签和原因。型号填写自己账户可以访问的型号，审计记录保存实际使用的标识和响应。

## 使用Web平台

Web平台的自然语言设计使用OpenAI规划模型。相应字段为：

| 用途 | 配置字段 |
| --- | --- |
| 规划密钥 | `OPENAI_API_KEY`，或当前账号在页面保存的OpenAI密钥 |
| 启用自然语言设计 | `POLYMER_AGENT_LIVE_DESIGN=1` |
| 设计型号 | `POLYMER_AGENT_DESIGN_MODEL`，默认`gpt-5.5` |
| 上层执行与修复规划型号 | `POLYMER_AGENT_UPPER_PLANNER_MODEL`，默认沿用设计型号 |
| DeepSeek审计型号 | `POLYMER_AGENT_AUDIT_DEEPSEEK_MODEL` |
| Qwen审计型号 | `POLYMER_AGENT_AUDIT_QWEN_MODEL` |
| GLM审计型号 | `POLYMER_AGENT_AUDIT_GLM_MODEL` |

平台审计密钥使用`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`和`ZHIPUAI_API_KEY`，也可以由当前账号在页面保存。环境中的密钥可在平台启动时填入管理员的空白密钥项；`POLYMER_AGENT_SEED_KEYS_FROM_ENV=0`关闭这种填入方式。

平台将账号密钥加密保存，接口返回配置状态。导入候选和本地校验可以先完成；自然语言设计检查规划密钥，真实审计检查三模型密钥，审计后的修复规划还使用规划密钥。

原MCP的`DEEPSEEK_MODEL`等字段与平台的`POLYMER_AGENT_AUDIT_*_MODEL`分别由各自入口读取。按选用入口配置相应字段即可。
