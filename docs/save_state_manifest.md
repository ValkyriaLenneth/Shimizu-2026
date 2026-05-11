# 保存现场清单

保存日期：2026-05-11

## Git 现场

本地保存现场提交：

```text
以 `git log --oneline -3` 显示的最新本地提交为准。
```

提交内容：

- YOLO 粗筛数据集构建、标注可视化、预测 review 脚本
- YOLOv9 源码副本
- Gemini 粗标注、补充标注、标签修正、可视化脚本
- YOLO 训练超参配置
- 客户汇报文档和 SVG 配图
- 数据集划分、环境准备、开发现场恢复指南
- `.gitignore`，排除数据、权重、训练产物、虚拟环境和本地归档

GitHub remote：

```text
origin  https://github.com/ValkyriaLenneth/Shimizu-2026.git
```

当前环境执行 `git push origin main` 时被 GitHub HTTPS 认证阻塞：

```text
fatal: could not read Username for 'https://github.com': No such device or address
```

因此代码已完成本地提交，并额外生成了 `git bundle`，待配置 GitHub 凭据后可继续 push。

后续推送命令：

```bash
cd /workspace/Shimizu-2026
git push origin main
```

## 归档包

归档目录：

```text
/workspace/Shimizu-2026/artifacts
```

数据集与训练结果包：

```text
artifacts/shimizu_2026_datasets_and_results_20260511.tar.gz
size: 14G
```

包含：

- `data/`
- `outputs/`
- `coarse_router_yolov9/datasets/`
- `coarse_router_yolov9/runs/`
- `coarse_router_yolov9/qa/`
- `coarse_router_yolov9/weights/`

代码快照包：

```text
artifacts/shimizu_2026_code_snapshot_20260511.tar.gz
size: 2.2M
```

Git bundle：

```text
artifacts/shimizu_2026_git_bundle_20260511.bundle
size: 2.2M
```

精确 SHA256 校验和写在：

```text
artifacts/SHA256SUMS
```

## 恢复入口

恢复指南：

```text
docs/recovery_guide.md
```

客户汇报文档：

```text
docs/client_report_yolo_coarse_router.md
```

恢复数据和训练结果：

```bash
cd /workspace/Shimizu-2026
tar -xzf artifacts/shimizu_2026_datasets_and_results_20260511.tar.gz
```

校验归档：

```bash
sha256sum artifacts/shimizu_2026_datasets_and_results_20260511.tar.gz
sha256sum artifacts/shimizu_2026_code_snapshot_20260511.tar.gz
sha256sum artifacts/shimizu_2026_git_bundle_20260511.bundle
```

从 bundle 恢复代码提交示例：

```bash
git clone artifacts/shimizu_2026_git_bundle_20260511.bundle restored-shimizu-2026
cd restored-shimizu-2026
git checkout main
```
