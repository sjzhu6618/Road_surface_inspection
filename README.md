# 多任务感知路面损伤多维量化评估系统（UI 重构版源码包）

这个压缩包是基于你当前 `RDD-YOLO-master` 项目整理出的第一阶段重构版源码工程，目标是把原来的命令行脚本逐步演进为可通过桌面界面操作的系统工程。

## 已包含

- `predict.py`：增强版推理脚本，支持图片 / 视频 / 摄像头 / RTSP、路面 ROI、视频跟踪去重、中文面板输出。
- `app/`：桌面端 UI 与服务层的第一阶段代码骨架。
- `launch_desktop.py`：桌面界面启动入口。
- `legacy/`：保留的旧训练、验证、检查脚本。
- `dataset/`：示例数据配置文件。
- `docs_implementation_guide.md`：改造实施说明。

## 还需要本地补充

为了控制压缩包体积，这个源码包没有附带大体积模型权重、完整数据集和 `depth_model`。请把这些资源放回项目根目录：

- `depth_model/`
- `runs/pothole_v1/weights/best.pt`（或你自己的权重）
- 测试图片 / 视频

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动桌面界面

```bash
python launch_desktop.py
```
