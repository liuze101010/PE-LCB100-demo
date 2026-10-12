# 面向聚合物分子动力学模拟的智能体框架

通过自然语言提出研究要求，由Codex调用本地科学工具，组织分子构建、模拟、轨迹分析、独立校验和结果归档。用户配置自己的模型与计算服务器。

默认分析结果为**密度、Rg、Ree、链形状、RDF和MSD**。不同聚合物通过材料适配器接入，分别登记结构表达、构建方法、力场、主链与端点定义和校验规则。

[使用指南](https://liuze101010.github.io/PE-LCB100-demo/) · [版本下载](https://github.com/liuze101010/PE-LCB100-demo/releases)

## 本地安装

使用Python 3.12，下载源码并解压，或运行：

```bash
git clone https://github.com/liuze101010/PE-LCB100-demo.git
cd PE-LCB100-demo
python -m venv .venv
```

Windows激活`.venv\Scripts\Activate.ps1`；Linux运行`source .venv/bin/activate`。在环境中安装并检查工具：

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

原MCP入口为`science_core/generated_code/md_analysis/mcp/pelcb_mcp_server.py`，提供59项工具。用Python和脚本的绝对路径配置Codex，通过`PELCB_PROJECT_CONFIG`指定个人项目JSON。配置示例见[examples/codex](examples/codex)，CLI和IDE入口见[官方MCP说明](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。

## 用自然语言发起任务

在源码目录启动Codex，先用`/mcp`确认`polymer_md`已连接，再说明材料、研究目标、条件和停止阶段：

> 使用已配置的服务器，研究支化聚乙烯的链结构与密度关系。先准备1个约C100的案例，采用项目登记的条件，输出密度、Rg、Ree、链形状、链间RDF和链质心MSD。请先核对结构、力场、分析定义和可用资源，并按案例保存实际产物。

新材料先检查配置，例如：

> 我希望研究聚醚的链构象和运动，重复单元为`*CCO*`。请检查构建器、力场及分析方法，列出需填写的聚合度、端基、链数和模拟条件，完成输入校验后停止。

不同材料使用对应适配器和分析定义。当前预置生产参数与物性参考规则用于聚乙烯；异原子主链、支化和共聚体系还需登记主链、端点及原子选区映射。

框架按“目标与输入—材料及环境检查—构建与校验—模拟—轨迹分析—结果检查与归档”组织任务。`AGENTS.md`规定原工具顺序、状态记录和用户指定的停止阶段，数值来自实际科学工具输出。

## 文件位置

- `science_core/`：构建、模拟、分析、校验、审计及处理源码。
- `docs/`：Codex、模型、服务器和材料接入说明。
- `examples/codex/`：Windows与Linux配置示例。
- `deploy/`：可选网页工作台与服务器入口。
- `tests/`：安装、接口、规则和接入检查。

部署ZIP由同一份源码生成：`python scripts/build_science_core_bundle.py`。可选网页工作台见[部署说明](deploy/README_DEPLOY.md)。Windows本地试用使用`POLYMER_AGENT_EXECUTOR_ENABLED=0`和`POLYMER_AGENT_COOKIE_SECURE=0`；Docker执行端使用PostgreSQL、任务队列和HTTPS。

## 核验与许可

安装与工具接入检查见[VALIDATION.md](VALIDATION.md)。GitHub Actions在Windows与Linux上运行自动检查；版本变化见[CHANGELOG.md](CHANGELOG.md)。

源码采用MIT许可。GROMACS、EMC和力场参数按各自条款安装，见[第三方软件说明](THIRD_PARTY_NOTICES.md)。个人密钥、服务器账户和计算结果保存在本地配置及任务目录中。
