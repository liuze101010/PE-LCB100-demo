# 面向聚合物分子动力学模拟的智能体框架

通过自然语言提出研究要求，由Codex调用本地科学工具，组织分子构建、模拟、轨迹分析、独立校验和结果归档。用户配置自己的模型与计算服务器。

默认分析结果为**密度、Rg、Ree、链形状、RDF和MSD**。不同聚合物使用对应的分子构建方法、力场、主链与端点定义和校验规则。

[使用指南](https://liuze101010.github.io/polymer-agent/) · [版本下载](https://github.com/liuze101010/polymer-agent/releases)

## 本地安装

使用Python 3.12，下载源码并解压，或运行：

```bash
git clone https://github.com/liuze101010/polymer-agent.git
cd polymer-agent
python -m venv .venv
```

Windows运行`.\.venv\Scripts\Activate.ps1`；Linux运行`source .venv/bin/activate`。在环境中安装并检查工具：

```bash
python -m pip install ".[science]"
python science_core/examples/minimal_validation.py
python -m polymer_agent_demo.scripts.doctor --stage local-validation --json
```

最小示例调用真实输入校验器，返回有效输入的`pass`和无效输入的`reject`。

## 连接工具与计算环境

| 连接内容 | 用途 | 说明 |
| --- | --- | --- |
| Codex与MCP | 理解自然语言要求，读取和调用科学工具 | [Codex接入](docs/CODEX_SETUP.md) |
| 模型账户 | Codex登录与独立审计API | [模型配置](docs/MODEL_SETUP.md) |
| SSH与科学软件 | 构建、模拟、监测和续算 | [服务器连接](docs/REMOTE_SETUP.md) |
| 材料适配器 | 结构、力场、分析定义与校验规则 | [材料配置](docs/MATERIAL_SETUP.md) |

MCP入口为`science_core/generated_code/md_analysis/mcp/pelcb_mcp_server.py`。用Python和脚本的绝对路径配置Codex，通过`PELCB_PROJECT_CONFIG`指定个人项目JSON。配置示例见[examples/codex](examples/codex)，CLI和IDE入口见[官方MCP说明](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。

## 用自然语言发起任务

在源码目录启动Codex，先用`/mcp`确认`polymer_md`已连接，再说明材料、研究目标、条件和停止阶段：

> 使用已配置的服务器，研究支化聚乙烯的链结构与密度关系。先准备1个约C100的案例，采用项目配置的条件，输出密度、Rg、Ree、链形状、链间RDF和链质心MSD。请先核对结构、力场、分析定义和可用资源，并按案例保存计算结果。

新材料先检查配置，例如：

> 我希望研究聚醚的链构象和运动，重复单元为`*CCO*`。请检查构建器、力场及分析方法，列出需填写的聚合度、端基、链数和模拟条件，完成输入校验后停止。

当前提供聚乙烯模拟方案。使用其他聚合物时，配置对应的构建方法、力场、模拟条件，以及分析所需的原子选择、链编号和端点定义。详见[材料配置](docs/MATERIAL_SETUP.md)。

框架按“目标与输入—材料及环境检查—构建与校验—模拟—轨迹分析—结果检查与归档”组织任务。`AGENTS.md`说明工具调用顺序、记录要求和停止阶段。

## 文件位置

- `science_core/`：构建、模拟、分析、校验、审计及处理源码。
- `docs/`：Codex、模型、服务器和材料接入说明。
- `examples/codex/`：Windows与Linux配置示例。
- `deploy/`：可选网页工作台与服务器入口。

可选网页工作台见[部署说明](deploy/README_DEPLOY.md)。Windows本地试用使用`POLYMER_AGENT_EXECUTOR_ENABLED=0`和`POLYMER_AGENT_COOKIE_SECURE=0`；Docker执行端使用PostgreSQL、任务队列和HTTPS。

## 许可

源码采用MIT许可。GROMACS、EMC和力场参数按各自条款安装，见[第三方软件说明](THIRD_PARTY_NOTICES.md)。个人密钥和服务器账户保存在本地配置中，计算结果保存在配置指定的本地或服务器任务目录。
