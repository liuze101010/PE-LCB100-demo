# PE-LCB100聚合物模拟智能体

本项目整理小论文《Validating LLM Agents for Reliable Polymer Simulations》的
原智能体代码，并提供论文案例交互回放。

流程为：自然语言目标 → 分子设计 → 构建与校验1 → 模拟与分析 → 校验2 →
审计与处理 → 再次校验 → 按案例归档。科学判断继续由原可执行规则完成。

## 查看演示

网页部署后的地址：https://liuze101010.github.io/PE-LCB100-demo/ 。

网页支持示例目标、案例搜索、步骤展开、处理前后对照，以及JSON、CSV和Markdown
下载。目标文字用于检索已有记录，页面标明“论文案例回放”。终态表包含100例，
92例PASS、8例REVIEW。保存的审计台账另有97个案例、291张模型票，按台账范围标注。
数据来源与对应代码可以逐例打开。

## 本机启动

使用Python 3.12。在Windows双击`start_demo.cmd`；Linux运行`bash run_demo.sh`。
启动脚本建立本项目的虚拟环境，安装演示依赖，并打开`/auto`。
手动安装方法见英文README。默认演示只读取随包记录，不读取个人密钥、服务器
或生产数据库。

## 试用科学工具

```bash
python -m pip install ".[science]"
python science_core/examples/minimal_validation.py
python -m polymer_agent_demo.scripts.doctor --stage local-validation --json
```

最小示例调用原输入校验器，输出有效输入的`pass`和无效输入的`reject`。
`science_core/`提供原构建、MD、分析、校验、审计、处理和MCP源码，以及配置、插件
和四阶段参数。远程科学核心ZIP从同一份公开源码生成。

原MCP入口是`science_core/generated_code/md_analysis/mcp/pelcb_mcp_server.py`。
填写独立本地配置，通过`PELCB_PROJECT_CONFIG`指定后启动。调用方法见源码中的MCP说明。

## 实际运行项目

```bash
python -m pip install ".[platform,science]"
```

Windows提供`start_platform.cmd`。手动启动时复制`.env.example`为本地`.env`，
填写配置，并显式设置`POLYMER_AGENT_APP_MODE=platform`。

自检：`python -m polymer_agent_demo.scripts.doctor --stage <阶段> --json`。
阶段可选`demo`、`local-validation`、`design`、`audit`、`md`和`all`，自检只读配置。

- 查看记录和本地输入校验使用本机环境。
- 分子设计配置规划模型密钥和型号。
- 审计配置登记的三模型及其密钥。
- 模拟配置个人计算服务器、EMC和GROMACS，原预检检查软件、权限与实时资源。

完整部署见[部署说明](deploy/README_DEPLOY.md)。模拟参数和科学规则按版本保存，
计算完成状态与科学验收状态分别记录。

## 开发和发布

开发安装：`python -m pip install -e ".[platform,science,dev]"`。
运行`python -m pytest -q`检查原功能和封装接口；
`python scripts/build_pages.py`生成`site/`，由GitHub Actions部署Pages。
实际验收结果见[VALIDATION.md](VALIDATION.md)。

当前发布整理聚乙烯小论文流程。原PI任务和文件保留在原工作目录。
源码采用MIT许可，第三方软件按自身许可安装。
