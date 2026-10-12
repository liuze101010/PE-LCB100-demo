# 演示与实际运行

使用说明统一见[README_CN.md](README_CN.md)。默认入口展示论文案例回放；
实际运行通过`start_platform.cmd`或显式`POLYMER_AGENT_APP_MODE=platform`启动。

公开网页由`python scripts/build_pages.py`生成，包含100例终态、案例筛选、八步流程、
原记录入口和数据下载。终态为92 PASS／8 REVIEW；97例／291票是保存的审计台账范围。

Windows双击`start_demo.cmd`，Linux运行`bash run_demo.sh`可看本机回放。
科学工具、模型和服务器配置按阶段准备，方法见README。
