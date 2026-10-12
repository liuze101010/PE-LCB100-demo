# 本地部署、服务器与模型连接

本说明用于把“面向聚合物分子动力学模拟的智能体框架”接到自己的工作目录、计算服务器和模型客户端。Codex中的规划模型选择步骤，执行智能体调用原MCP提供的构建、校验、资源检查、模拟和分析工具。Codex连接方法见[连接说明](CODEX_SETUP.md)。

## 1. 安装科学工具

在自己的仓库副本中使用Python 3.12创建虚拟环境，安装科学工具：

```bash
python -m pip install -e ".[science]"
python science_core/examples/minimal_validation.py
python -m polymer_agent_demo.scripts.doctor --stage local-validation --json
```

最小示例只调用输入校验器。远程计算另需个人SSH账户、EMC、GROMACS 2024.4及站点认可的计算资源。Web平台的Postgres、Redis、HTTPS和worker部署见[部署说明](../deploy/README_DEPLOY.md)；直接使用原MCP工具可以独立于Web平台部署。

## 2. 准备个人项目配置

以[公开配置](../science_core/generated_code/md_analysis/configs/pelcb100.public.json)为模板，在自己的任务目录保存本地JSON。个人主机、账户、配置和产物保存在个人目录。

配置加载器支持`${VARIABLE}`环境变量，并递归展开JSON中的字符串。本地相对路径以`science_core/`为基准；放在仓库外的配置建议使用本地绝对路径。

| 字段 | 填写内容 |
| --- | --- |
| `project_id` | 案例批次编号，用于组织清单和汇总记录 |
| `manifest_csv` | 自己的候选清单CSV |
| `case_roots` | 实际案例输入目录；至少保留一个含`label`和`path`的条目 |
| `bridge_roots` | 已完成EMC→GROMACS转换的输入目录 |
| `integrated_root` | 分析汇总目录 |
| `runtime_output_dir`、`supervisor_output_dir`、`emc_output_dir` | 资源探测、调度和构建记录目录 |
| `a3_results_dir`、`repair_loop_dir`、`warn_audit_output_dir` | 校验、处理和审计记录目录 |
| `packing_void_output_dir`、`overlay_dir`、`rule_profile_dir` | 堆积分析、处理结果和科学规则目录 |
| `o7_queue_dir`、`o7_closure_dir` | 原O7队列及后处理记录目录 |
| `new_server`、`newnew_server`、`old_server` | 对应SSH连接入口，通常填写`用户名@主机名` |
| `remote_md_roots` | 对应服务器上的实际MD目录 |
| `emc_remote_root`、`old_emc_root` | 远程构建输出目录和EMC安装目录 |
| `old_server_postprocess_root`、`old_server_input_root` | 使用相应后处理入口时的远程输出和输入目录 |
| `new_gmx`、`newnew_gmx` | 对应服务器上的GROMACS 2024.4可执行文件 |

下面的配置生成示例保留现有字段，把可配置的本地产物放入一个独立目录。Windows示例目录为`C:/polymer-agent-local`；Linux用户将`local_root`改为自己的可写绝对目录。

```python
import json
from pathlib import Path

repo = Path.cwd()  # 在自己的仓库根目录执行
local_root = Path("C:/polymer-agent-local")
local_root.mkdir(parents=True, exist_ok=True)
template = repo / "science_core/generated_code/md_analysis/configs/pelcb100.public.json"
config = json.loads(template.read_text(encoding="utf-8"))
config["manifest_csv"] = str(local_root / "inputs/manifest.csv")
config["case_roots"] = [{"label": "personal_cases", "path": str(local_root / "inputs/cases")}]
config["bridge_roots"] = [str(local_root / "inputs/bridge")]
for field in (
    "integrated_root", "runtime_output_dir", "supervisor_output_dir",
    "emc_output_dir", "a3_results_dir", "repair_loop_dir",
    "warn_audit_output_dir", "packing_void_output_dir", "overlay_dir",
    "rule_profile_dir", "o7_queue_dir", "o7_closure_dir",
):
    config[field] = str(local_root / "results" / field)
destination = local_root / "project.json"
destination.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
print(destination)
```

这个示例创建配置文件。将真实清单、案例和转换后的输入放入相应目录，再填写服务器字段。MCP自身的工具账本和准备文件仍使用原服务的`md_analysis/output/pelcb_mcp_sidecar/`目录。

## 3. 填写服务器与资源字段

公开配置中的`${REMOTE_USER}`、`${REMOTE_HOST}`、`${PELCB100_REMOTE_HOME}`和`${PELCB100_REMOTE_ROOT}`分别表示自己的SSH用户名、主机名和远程可写目录。可以在MCP进程环境中提供这些值，也可以在本地JSON中填写实际值。

`cluster_profiles`包含`production_new`、`production_newnew`和`o7_clean`。按所用入口填写相应配置，原O7提交入口使用`o7_clean`。

| 集群配置字段 | 填写内容 |
| --- | --- |
| `host`、`host_label` | SSH入口和对应的`new`/`newnew`标签 |
| `remote_root` | 本批任务的远程可写目录 |
| `gmx_bin`、`engine_version_expected` | GROMACS可执行文件和`GROMACS 2024.4`版本要求 |
| `engine_module`、`module_preset` | 站点已有的软件初始化；可执行文件已在PATH时使用公开配置的`none` |
| `allowed_nodes`、`disallowed_nodes` | 本批可以使用和应跳过的节点 |
| `default_threads`、`threads_by_node` | 每例线程数及节点覆盖值 |
| `node_cap_default`、`node_cap_by_node` | 节点并发上限 |
| `per_gpu_target_default`、`per_gpu_target_by_node` | 每GPU并发上限；初次接入可以从每卡一例开始 |
| `launch_via_node` | 是否从登录入口转到指定计算节点执行 |

这些字段限定任务可用资源，原资源探针继续读取当次CPU/GPU占用。实际节点、GPU和线程分配依据当次探测结果。

先在自己的终端核对SSH主机指纹，并验证免密SSH和远程软件。示例中的`your-user@cluster.example.org`是占位入口，填写自己的服务器后再执行：

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes your-user@cluster.example.org hostname
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes your-user@cluster.example.org 'gmx --version'
```

原MCP把本次加载的配置绝对路径传给O7提交、处理循环和NVT续算入口，子进程同时继承对应的`PELCB_PROJECT_CONFIG`与`LCB_PROJECT_CONFIG`。后处理closure、MSD拟合、overlay和堆积分析通过各自已有的队列、输入、输出路径参数使用本项目目录。

专用输入另行准备：O7来源注册表位于`supervisor_output_dir/o7_clean_engine_audit_20260522/source_dirs_combined_unique.csv`；MSD时间序列目录、overlay候选文件和NVT重新拟合分类使用相应原CLI参数。原closure还保留专用主机和引擎fallback，个人接入时需核对案例记录中的`target_host`/`source_host`、远程路径与实际引擎。直接调用原子脚本时，先读取其`--help`并确认本次输入。

## 4. 连接原MCP服务

在MCP客户端设置Python和脚本的绝对路径，并向服务进程传入本地配置路径。例如Windows环境变量：

```powershell
$env:PELCB_PROJECT_CONFIG = "C:/polymer-agent-local/project.json"
```

Linux环境变量示例：

```bash
export PELCB_PROJECT_CONFIG=/your/local/run-directory/project.json
```

服务命令为：

```bash
python science_core/generated_code/md_analysis/mcp/pelcb_mcp_server.py
```

服务通过stdio等待客户端请求。客户端先完成握手和`tools/list`，然后读取工具的实际`inputSchema`。先调用以下本地检查工具即可核对连接：

1. `list_polymer_preparation_capabilities`：查看已登记构建器、材料适配器和力场。
2. `validate_design_candidate`：使用`branch_points=1`、`branch_carbons=8`、`branch_position_pattern="mid"`验证PE-LCB最小输入。
3. `validate_polymer_design_candidate`或`validate_or_plan_polymer_input`：检查自己的结构输入及构建计划。
4. `list_supported_rule_properties`和`get_rule_profile`：查看所需性质和现有验收规则。

工具列表和本地输入校验可以在模型密钥尚未配置时完成。先通过这些步骤确认Python环境、MCP连接和输入校验接口。

## 5. 配置模型

规划模型由Codex或其他MCP客户端配置。原MCP的真实三模型审计另外从服务进程环境读取：

| 用途 | 密钥环境变量 | 原MCP型号环境变量 | 保留的默认型号 |
| --- | --- | --- | --- |
| DeepSeek审计 | `DEEPSEEK_API_KEY` | `DEEPSEEK_MODEL` | `deepseek-v4-pro` |
| Qwen审计 | `DASHSCOPE_API_KEY` | `QWEN_MODEL` | `qwen3.7-max` |
| GLM审计 | `ZHIPUAI_API_KEY` | `GLM_MODEL` | `glm-4.6` |

真实审计使用`use_mock_clients=false`，并按所用账户配置可访问的型号。密钥通过个人环境或客户端支持的密钥注入方式提供，Git仓库保存环境变量名称和空白示例。

Web平台使用`POLYMER_AGENT_AUDIT_DEEPSEEK_MODEL`、`POLYMER_AGENT_AUDIT_QWEN_MODEL`和`POLYMER_AGENT_AUDIT_GLM_MODEL`选择审计型号；这些字段属于平台封装。直接使用原MCP时采用上表字段。

## 6. 用自然语言交给Codex

可以先给Codex这样的任务：

> 按已登记工具检查这个聚合物任务的结构、构建方法和力场。先读取相关资料并保存设计依据，完成校验1后停止。请报告配置缺项、实际工具返回和案例目录。

已有输入和服务器配置准备完成后，在新的计算任务中明确研究条件、目标性质、案例范围、计算预算和停止阶段。Codex选择登记步骤，调用原工具，并按案例保存输入、参数、产物和状态。

现有工具调用顺序为：资料与设计 → 输入校验与构建 → 资源检查与模拟 → 分析与规则校验 → WARN审计 → 登记处理 → 同规则再次校验 → 归档。不同工具的实际作用如下：

| 工具或参数 | 实际作用 |
| --- | --- |
| `validate_*`、能力列表 | 本地输入、注册能力和构建计划检查 |
| `materialize_polymer_input` | 写出本地EMC输入及案例元数据 |
| `plan_supervisor_slots(refresh=false)` | 读取已有资源记录 |
| `plan_supervisor_slots(refresh=true)` | 调用原资源探针，可能执行SSH检查 |
| `discover_rule_evidence(use_live=false)` | 使用本地结构化依据 |
| `discover_rule_evidence(use_live=true)` | 请求登记的外部资料源 |
| `run_warn_audit_diagnosis(execute=false)` | 预览当前记录的规则诊断 |
| `run_warn_audit_diagnosis(execute=true)` | 在对应执行约定满足后调用审计模型 |
| `submit_supervisor_case`、`submit_md_clean_engine` | 在对应执行约定满足后提交原MD任务 |
| `run_postprocess_closure`、`run_postprocess_repair`、`extend_clean_nvt_prod` | 执行相应分析、处理或续算，可能访问服务器 |
| `plan_repair_loop`等计划入口 | 调用原计划脚本并写本地计划记录；按具体签名检查作用和产物 |

改变状态的工具要求`execute=true`、登记的工具项目标识、匹配的`allowlist_tag`和至少12个字符的具体`reason`。配置中的批次`project_id`与这些工具项目标识分别承担批次组织和工具授权职责。

例如输入准备使用`polymer_md_preparation`与`POLYMER_MCP_EXECUTE_PREPARATION`；O7提交使用`o7clean`与`PELCB_MCP_EXECUTE_O7CLEAN`；真实WARN审计使用`pelcb_repair_loop`与`PELCB_WARN_AUDIT_EXECUTE`。具体以工具返回的执行要求和源码签名为准。

已有聚乙烯熔体示例采用PE-LCB C100/C101、OPLS-AA、400 K、1 atm和150链协议。通用输入适配器覆盖更多结构表示；新材料的模拟适配和物性验收使用相应登记。规则缺项和证据缺口保留为WARN/REVIEW，案例结果同时记录计算完成情况和科学验收状态。材料字段和分析条件见[材料配置说明](MATERIAL_SETUP.md)，模型字段见[模型连接说明](MODEL_SETUP.md)。
