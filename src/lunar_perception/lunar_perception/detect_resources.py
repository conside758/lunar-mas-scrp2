"""资源探测服务：给定位姿与半径，返回范围内的资源探测结果（Stage1 简化，带少量噪声）。"""
import math
import random
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger
from lunar_msgs.msg import ResourceDetection
from lunar_msgs.srv import DetectResources


class DetectResourcesNode(Node):
    def __init__(self):
        super().__init__('detect_resources')
        self.declare_parameter('scenario_file', '')
        self.scenario_file = self.get_parameter('scenario_file').value
        self.scenario = None
        if self.scenario_file:
            self.load_scenario()
        self.srv = self.create_service(DetectResources, 'detect_resources', self.detect_cb)

    def load_scenario(self):
        with open(self.scenario_file, 'r', encoding='utf-8') as f:
            self.scenario = yaml.safe_load(f)

    def detect_cb(self, request, response):
        if self.scenario is None:
            self.load_scenario()
        cx = request.center.position.x
        cy = request.center.position.y
        for r in self.scenario['resources']:
            dx = r['x'] - cx
            dy = r['y'] - cy
            if math.hypot(dx, dy) <= request.radius:
                d = ResourceDetection()
                d.zone_id = int(r['id'])
                d.pose.position.x = float(r['x'])
                d.pose.position.y = float(r['y'])
                noise = 1.0 + random.uniform(-0.1, 0.1)
                d.estimated_amount = float(r['amount']) * noise
                d.confidence = 0.9
                response.detections.append(d)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DetectResourcesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
