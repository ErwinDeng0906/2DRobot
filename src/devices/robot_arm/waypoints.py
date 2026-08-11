"""命名点位管理: 记录/读取机械臂关键位姿 (关节角 + TCP位姿)。

点位存为 json: {name: {joints:[6], tcp:[6], ts:..}}。
关节角是回放主依据(无IK歧义); tcp 仅供参考/视觉对齐。
"""
import json
import os
import time


class WaypointStore:
    def __init__(self, path):
        self.path = path
        self.points = {}
        if os.path.exists(path):
            self.load()

    def load(self):
        with open(self.path, encoding="utf-8") as f:
            self.points = json.load(f)
        return self

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.points, f, indent=2, ensure_ascii=False)
        return self

    def record(self, name, joints, tcp):
        self.points[name] = {
            "joints": list(joints),
            "tcp": list(tcp),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return self

    def get_joints(self, name):
        return self.points[name]["joints"]

    def get_tcp(self, name):
        return self.points[name]["tcp"]

    def has(self, name):
        return name in self.points

    def names(self):
        return list(self.points.keys())
