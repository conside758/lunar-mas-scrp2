"""独立启动 Gazebo 与通用世界（用于手动调试）。

用法：
    ros2 launch lunar_gazebo gazebo.launch.py gui:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_gazebo = get_package_share_directory('lunar_gazebo')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')
    world_file = os.path.join(pkg_gazebo, 'worlds', 'lunar_field.world')

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('gui', default_value='false'))

    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')]),
        launch_arguments={
            'world': world_file,
            'gui': LaunchConfiguration('gui'),
            'verbose': 'true',
        }.items(),
    ))
    return ld
