"""将资源区与卸载点发布为 rviz2 MarkerArray，便于可视化。"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from lunar_msgs.msg import ResourceArray


class ResourceMarkers(Node):
    def __init__(self):
        super().__init__('resource_markers')
        self.pub = self.create_publisher(MarkerArray, 'resource_markers', 10)
        self.create_subscription(ResourceArray, 'resources', self.res_cb, 10)
        self.create_subscription(PoseStamped, 'depot', self.depot_cb, 10)
        self.markers = []

    def res_cb(self, msg: ResourceArray):
        arr = MarkerArray()
        for i, z in enumerate(msg.zones):
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'resources'
            m.id = z.id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose = z.pose
            m.pose.position.z = 0.15
            m.scale.x = 0.8
            m.scale.y = 0.8
            m.scale.z = 0.3
            m.color = ColorRGBA(r=0.2, g=0.8, b=0.3, a=0.7)
            arr.markers.append(m)
        self.pub.publish(arr)

    def depot_cb(self, msg: PoseStamped):
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'depot'
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose = msg.pose
        m.pose.position.z = 0.2
        m.scale.x = 0.8
        m.scale.y = 0.8
        m.scale.z = 0.4
        m.color = ColorRGBA(r=0.95, g=0.85, b=0.2, a=0.8)
        arr = MarkerArray()
        arr.markers.append(m)
        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ResourceMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
