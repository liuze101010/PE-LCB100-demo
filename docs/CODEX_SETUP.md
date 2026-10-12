# 在本地连接Codex

Codex根据自然语言要求调用工具，本项目的MCP服务提供科学计算接口。MCP在本机运行，服务器计算通过用户配置的SSH连接执行。

## 安装工具

使用Python 3.12，在源码目录创建并激活`.venv`，执行：

```bash
python -m pip install ".[science]"
python science_core/examples/minimal_validation.py
```

新建`configs/`目录，复制`science_core/generated_code/md_analysis/configs/pelcb100.public.json`为`configs/project.local.json`，填写个人输入、输出、服务器及科学软件路径。路径基准和字段见[服务器连接](REMOTE_SETUP.md)。

## 登录Codex

按[官方CLI指南](https://learn.chatgpt.com/docs/codex/cli)安装Codex CLI，或安装Codex IDE扩展。CLI登录：

```bash
codex login
codex login status
```

Codex支持ChatGPT账户或API登录，见[官方认证说明](https://learn.chatgpt.com/docs/auth)。Codex登录用于其自身模型调用，独立审计API按[模型配置](MODEL_SETUP.md)提供。

## 登记MCP服务

采用[Windows配置](../examples/codex/windows.toml)或[Linux配置](../examples/codex/linux.toml)，替换实际项目绝对路径，将相应表加入个人`~/.codex/config.toml`，保留原有其他配置。

示例包含Python解释器、MCP脚本、工作目录和项目JSON。`env_vars`只列变量名，实际模型密钥在个人环境中设置。

CLI也可登记入口。以下Windows示例需先替换路径：

```powershell
codex mcp add polymer_md --env PELCB_PROJECT_CONFIG=C:/Projects/polymer-md-agent/configs/project.local.json -- C:/Projects/polymer-md-agent/.venv/Scripts/python.exe C:/Projects/polymer-md-agent/science_core/generated_code/md_analysis/mcp/pelcb_mcp_server.py
codex mcp list
```

Linux使用实际`.venv/bin/python`和项目路径。CLI添加后，按示例补入`cwd`及审计所需的`env_vars`。CLI与IDE共享配置；IDE的MCP设置可添加STDIO服务，保存后重启扩展。[官方MCP说明](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)

## 输入自然语言任务

从源码目录启动`codex`，或在IDE中打开源码文件夹。用`/mcp`确认`polymer_md`已连接，先检查环境：

> 请读取项目配置和工具定义，检查Python、SSH、GROMACS、EMC及材料适配器，列出需补齐的项。本次只检查环境与输入。

准备计算时说明材料、结构、条件和范围：

> 使用登记的聚乙烯构建器准备1个约C100的支化案例，沿用项目条件，输出密度、Rg、Ree、链形状、RDF和MSD。请完成构建、校验、提交、监测和结果检查，并按案例保存实际产物。

新材料先检查适配，例如：

> 我希望研究聚酰亚胺的链构象与运动。请检查单体和重复单元表达、构建器、力场参数、主链与端点映射和分析方法，整理配置并通过输入校验后停止。

`AGENTS.md`提供工具顺序和记录要求。任务可以指定只准备输入、只构建、接续已有模拟或处理已有轨迹。

## 接入问题

| 表现 | 检查位置 |
| --- | --- |
| MCP启动失败 | Python、脚本绝对路径与科学依赖 |
| 项目JSON未找到 | `PELCB_PROJECT_CONFIG`及本地文件 |
| 审计无法调用模型 | 环境变量与MCP的`env_vars` |
| SSH连接失败 | SSH配置、公钥、主机指纹和账户 |
| 材料缺少分析映射 | [材料配置](MATERIAL_SETUP.md)中的主链、端点与选区 |
