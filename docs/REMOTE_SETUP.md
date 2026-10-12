# 本地部署与服务器连接

用户在本地安装科学工具，通过MCP连接Codex，再配置自己的计算服务器。模型账户见[模型连接](MODEL_SETUP.md)，Codex操作见[连接说明](CODEX_SETUP.md)。

## 安装与检查

使用Python 3.12，在源码目录创建并激活虚拟环境，运行：

```bash
python -m pip install -e ".[science]"
python science_core/examples/minimal_validation.py
python -m polymer_agent_demo.scripts.doctor --stage local-validation --json
```

最小示例检查分子输入。远程计算还需要SSH账户、EMC、GROMACS 2024.4和可用计算资源。MCP可以独立部署；可选网页工作台见[部署说明](../deploy/README_DEPLOY.md)。

## 填写项目配置

复制[配置模板](../science_core/generated_code/md_analysis/configs/pelcb100.public.json)到自己的任务目录，填写输入、输出和服务器信息。通过`PELCB_PROJECT_CONFIG`指定这个JSON文件。

配置支持`${VARIABLE}`环境变量。本地相对路径以`science_core/`为基准；仓库外的任务建议填写绝对路径。

| 字段 | 填写内容 |
| --- | --- |
| `project_id` | 本批案例编号 |
| `manifest_csv` | 候选清单CSV |
| `case_roots` | 案例输入目录，条目包含`label`和`path` |
| `bridge_roots` | EMC转换为GROMACS格式后的输入目录 |
| `integrated_root` | 分析汇总目录 |
| `runtime_output_dir`、`supervisor_output_dir`、`emc_output_dir` | 资源检查、任务调度和构建记录目录 |
| `a3_results_dir`、`repair_loop_dir`、`warn_audit_output_dir` | 校验、处理和审计记录目录 |
| `packing_void_output_dir`、`overlay_dir`、`rule_profile_dir` | 堆积分析、结果整理和校验规则目录 |
| `o7_queue_dir`、`o7_closure_dir` | 模拟队列和后处理记录目录 |
| `new_server`、`newnew_server`、`old_server` | 相应SSH入口，通常为`用户名@主机名` |
| `remote_md_roots` | 各服务器上的模拟目录 |
| `emc_remote_root`、`old_emc_root` | 远程构建目录和EMC安装目录 |
| `old_server_postprocess_root`、`old_server_input_root` | 相应后处理工具的远程输出和输入目录 |
| `new_gmx`、`newnew_gmx` | 相应服务器上的GROMACS可执行文件 |

把真实清单、案例和转换后的输入放入配置目录。接续已有模拟时，填写实际运行目录和案例编号。

## 配置服务器资源

模板中的`${REMOTE_USER}`、`${REMOTE_HOST}`、`${PELCB100_REMOTE_HOME}`和`${PELCB100_REMOTE_ROOT}`分别填写SSH用户名、主机名及远程可写目录，也可以由进程环境提供。

`cluster_profiles`包含`production_new`、`production_newnew`和`o7_clean`。按所用提交工具填写对应项；`submit_md_clean_engine`使用`o7_clean`。

| 字段 | 填写内容 |
| --- | --- |
| `host`、`host_label` | SSH入口及对应的`new`/`newnew`标签 |
| `remote_root` | 本批任务的远程目录 |
| `gmx_bin`、`engine_version_expected` | GROMACS路径和`GROMACS 2024.4`版本要求 |
| `engine_module`、`module_preset` | 服务器的软件初始化方式；软件已在PATH时填`none` |
| `allowed_nodes`、`disallowed_nodes` | 可用节点与应跳过的节点 |
| `default_threads`、`threads_by_node` | 每例线程数及节点设置 |
| `node_cap_default`、`node_cap_by_node` | 每节点并发上限 |
| `per_gpu_target_default`、`per_gpu_target_by_node` | 每张GPU并发上限；初次使用可设为1 |
| `launch_via_node` | 是否从登录入口转到计算节点执行 |

工具在提交前检查CPU/GPU实时占用，再分配节点、GPU和线程。先在自己的终端确认主机指纹，验证免密SSH和科学软件。替换下列占位账户后运行：

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes your-user@cluster.example.org hostname
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes your-user@cluster.example.org 'gmx --version'
```

## 连接MCP

客户端使用Python解释器和脚本的绝对路径，并传入个人配置。例如Windows：

```powershell
$env:PELCB_PROJECT_CONFIG = "C:/polymer-agent-local/project.json"
```

Linux：

```bash
export PELCB_PROJECT_CONFIG=/your/local/run-directory/project.json
```

MCP启动命令：

```bash
python science_core/generated_code/md_analysis/mcp/pelcb_mcp_server.py
```

服务通过stdio接收客户端请求。Codex连接后，先读取工具列表和输入要求，检查：

| 工具 | 用途 |
| --- | --- |
| `list_polymer_preparation_capabilities` | 查看构建方法、材料适配器和力场 |
| `validate_polymer_design_candidate`、`validate_or_plan_polymer_input` | 检查结构输入及构建计划 |
| `list_supported_rule_properties`、`get_rule_profile` | 查看性质与校验依据 |
| `plan_supervisor_slots(refresh=true)` | 通过SSH检查可用资源 |

这些连接与输入检查可以先于模型API配置完成。实际计算时，Codex按工具要求提供`execute`、项目标识、`allowlist_tag`和操作原因，再调用构建、提交、分析及处理工具。

## 处理已有任务

提交、续算和分析使用同一项目配置。接续前让Codex核对案例编号、运行目录、进程和日志。

| 任务 | 需要准备 |
| --- | --- |
| 模拟后处理 | 案例来源清单、队列记录、实际服务器和GROMACS路径 |
| MSD拟合 | MSD时间序列目录及采样信息 |
| 结果整理 | 待整理案例清单及输入、输出目录 |
| NVT续算与重新拟合 | 案例分类记录、TPR、检查点和输出目录 |

专用工具的参数由其MCP说明或CLI的`--help`列出。使用后处理工具时，核对案例记录中的`target_host`、`source_host`及远程路径。

## 用自然语言发起任务

> 请检查这个聚合物任务的结构、构建方法、力场和服务器环境，完成输入校验后停止，并报告需要补充的信息。

准备完成后，说明研究条件、目标性质、案例范围和计算预算。框架按“设计与输入—构建与校验—资源检查与模拟—分析与校验—异常处理—再次校验—归档”组织任务。

当前预置聚乙烯熔体示例采用C100/C101、OPLS-AA、400 K、1 atm和150条链。其他聚合物配置对应的构建、模拟和分析方法，见[材料说明](MATERIAL_SETUP.md)。
