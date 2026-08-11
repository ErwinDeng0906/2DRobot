"""SCARA 送检流水线（视觉伺服取放）。

不依赖硬件的骨架：硬件抽象在 backends，检测在 ../vision/wafer_detect，
伺服在 ../vision/servo，标定加载在 ../calib/loaders，取放序列在 ../sequence。
架构见 docs/scara_pipeline_architecture.md。
"""
