# 网页工作台部署

本页介绍可选的多用户网页工作台。使用本地Codex时，按[Codex指南](../docs/CODEX_SETUP.md)连接MCP即可。

## 启动工作台

Linux服务器需要Docker Engine和Compose插件。复制`.env.example`为`.env`，填写配置，并将`POLYMER_AGENT_DOMAIN`设为可通过HTTPS访问的域名。

```bash
docker compose build
docker compose up -d
docker compose exec api python -m polymer_agent_demo.scripts.doctor --stage md --json
```

首次登录使用`POLYMER_AGENT_BOOTSTRAP_ADMIN`。管理员创建个人账号后，删除`.env`中的初始密码并重启API和worker。

Windows本地试用设置`POLYMER_AGENT_EXECUTOR_ENABLED=0`和`POLYMER_AGENT_COOKIE_SECURE=0`。计算服务使用PostgreSQL、Redis、worker及HTTPS。

## 连接计算服务器

1. 用户在页面创建服务器连接。
2. 工作台生成SSH公钥，用户将其加入服务器账号的`~/.ssh/authorized_keys`。
3. 管理员确认主机、分区和任务目录。
4. 用户选择自动安装工具、服务器现有软件环境或Apptainer/Singularity镜像，运行环境检查。
5. 用户填写模型密钥，再提出自然语言任务。

自动安装使用EMC 9.4.4（`emc-pypi==2025.8.21`）及GROMACS 2024.4，工具安装在个人任务目录。连接通过软件、SSH主机指纹、Slurm和目录权限检查后，状态变为`ready`。模型配置见[说明](../docs/MODEL_SETUP.md)。

用户的SSH私钥与API密钥加密保存，页面显示配置状态。

## 服务器要求

- Slurm命令：`sbatch`、`squeue`、`sacct`和`scancel`。
- Python 3、EMC、GROMACS及力场参数。
- 自动编译安装另需CMake、GNU Make和C++17编译器；GPU计算需兼容的CUDA工具与驱动。
- 用户对任务目录有写入权限。

也可以选择服务器已有的软件模块或容器。需要手动安装计算入口时，将`deploy/remote_runner.py`和`deploy/install_remote_runner.sh`上传服务器，再运行：

```bash
bash install_remote_runner.sh /absolute/path/to/site-pelcb-pipeline
polymer-agent-remote self-test --json
```

自有服务器计算脚本可通过`POLYMER_AGENT_PRODUCTION_PIPELINE`配置。计算入口按构建、模拟和后处理分阶段运行：

```bash
pelcb-pipeline-core self-test --json
polymer-agent-remote stage --stage build --project-spec project.json \
  --manifest manifest.json --work-root work
polymer-agent-remote stage --stage md --project-spec project.json \
  --manifest manifest.json --work-root work
polymer-agent-remote stage --stage postprocess --project-spec project.json \
  --manifest manifest.json --work-root work
```

## 提交计算

用户确认CPU、GPU、最长运行时间和处理次数。服务器连接为`ready`、项目检查通过、Redis与worker正常运行，并设置`POLYMER_AGENT_EXECUTOR_ENABLED=1`后，可以提交任务。

工作台按案例保存计算结果、校验状态及处理记录，大型轨迹保留在服务器。用户可以指定准备输入、分子构建、模拟或轨迹分析作为本次停止阶段。

## 配置不同聚合物

当前预置聚乙烯示例采用OPLS-AA、400 K、1 atm和150条链。其他聚合物需配置对应的分子构建、力场、模拟条件、主链与端点、分析选区和校验依据，见[材料配置](../docs/MATERIAL_SETUP.md)。

构建后，EMC和GROMACS `grompp`检查原子类型及拓扑。缺少参数时，用户补齐参数再继续；需要补充科学依据的结果保留相应校验状态。
