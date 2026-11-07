# 语义分割

- course-source：DeepLabV3+ 课程代码（保留原 MIT 许可）。
- uavid_exp：各轮训练、按序列验证、误差分析和推理代码；第八轮入口为 uavid_exp/v8_pipeline.py，单图及最终推理见 v8_predict.py。
- v8 使用的数据、分折索引、先前各轮检查点、模型预训练权重和测试标签未包含；脚本中的 /data/aerospace-intelligence 是原运行时目录，需自行适配并复核依赖。不要在缺少冻结检查点时声称可复现相同数值。
- UAVid 官网：https://uavid.nl/ ；DINOv2：https://github.com/facebookresearch/dinov2 ；外部 AeroScapes 单折尝试：https://github.com/ishann/aeroscapes 。
- 测试标签来自公开镜像，历史已被多轮开发使用，测试结果仅作描述性对照。
