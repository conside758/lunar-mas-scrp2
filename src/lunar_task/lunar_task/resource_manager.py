"""资源管理器：维护资源区权威状态，发布资源/卸载点，提供重置服务。"""
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from lunar_msgs.msg import ResourceZone, ResourceArray


class ResourceManager(Node):
    def __init__(self):
        super().__init__('resource_manager')
        self.declare_parameter('scenario_file', '')
        self.scenario_file = self.get_parameter('scenario_file').value
        if not self.scenario_file:
            self.get_logger().error('scenario_file parameter not set')
            raise RuntimeError('scenario_file parameter required')

        self.scenario = None
        self.load_scenario()

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.res_pub = self.create_publisher(ResourceArray, 'resources', latched)
        self.depot_pub = self.create_publisher(PoseStamped, 'depot', latched)
        self.reset_srv = self.create_service(Trigger, 'reset_scenario', self.reset_cb)
        self.publish()
        self.get_logger().info(
            f'ResourceManager ready: {len(self.scenario["resources"])} resources, '
            f'depot=({self.scenario["depot"]["x"]},{self.scenario["depot"]["y"]})')

    def load_scenario(self):
        with open(self.scenario_file, 'r', encoding='utf-8') as f:
            self.scenario = yaml.safe_load(f)

    def publish(self):
        arr = ResourceArray()
        for r in self.scenario['resources']:
            z = ResourceZone()
            z.id = int(r['id'])
            z.pose.position.x = float(r['x'])
            z.pose.position.y = float(r['y'])
            z.pose.position.z = 0.0
            z.amount = float(r['amount'])
            z.value = float(r.get('value', 1.0))
            arr.zones.append(z)
        self.res_pub.publish(arr)

        depot = self.scenario['depot']
        dp = PoseStamped()
        dp.header.stamp = self.get_clock().now().to_msg()
        dp.header.frame_id = 'world'
        dp.pose.position.x = float(depot['x'])
        dp.pose.position.y = float(depot['y'])
        dp.pose.position.z = 0.0
        self.depot_pub.publish(dp)

    def reset_cb(self, request, response):
        self.load_scenario()
        self.publish()
        response.success = True
        response.message = 'scenario reset'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ResourceManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
