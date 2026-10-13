"""录制 Gazebo 相机画面为 JPG 帧序列（供 OpenCV 拼成 MP4）。

订阅 /demo_cam/image_raw（sensor_msgs/Image），转 BGR 存 JPG，按 --duration 时长录制，
到点自动退出。无头环境（软件渲染）+ 无 ffmpeg，故存帧后用 OpenCV VideoWriter 拼视频。

用法：ros2 run lunar_env camera_recorder <out_dir> <duration_s>
"""
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraRecorder(Node):
    def __init__(self, out_dir, duration):
        super().__init__('camera_recorder')
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.duration = duration
        self.t0 = time.time()
        self.count = 0
        self.sub = self.create_subscription(Image, '/demo_cam/camera1/image_raw', self.cb, 10)
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info(f'recording to {out_dir} for {duration}s')

    def cb(self, msg: Image):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if 'rgb8' in msg.encoding.lower():
                bgr = arr[:, :, ::-1].copy()  # RGB -> BGR
            else:
                bgr = arr.copy()
            path = os.path.join(self.out_dir, f'frame_{self.count:06d}.jpg')
            cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            self.count += 1
        except Exception as e:
            self.get_logger().error(f'frame error: {e}')

    def tick(self):
        if time.time() - self.t0 > self.duration:
            self.get_logger().info(f'finished, saved {self.count} frames to {self.out_dir}')
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    out_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/demo_frames'
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 135.0
    node = CameraRecorder(out_dir, duration)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
