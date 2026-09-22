"""轻量指标记录：按行累积 dict 并落盘 CSV。"""
import csv
import os


class MetricsLogger:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def record(self, **kwargs):
        self.rows.append(kwargs)

    def save(self):
        if not self.rows:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
