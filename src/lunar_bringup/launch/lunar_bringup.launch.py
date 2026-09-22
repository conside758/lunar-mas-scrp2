"""顶层 launch：启动 Gazebo、三台机器人（模型/控制器/步态）、资源管理器与可视化。

架构说明：
- Gazebo 通过 gazebo_ros 的 gazebo.launch.py 启动（正确设置 GAZEBO_PLUGIN_PATH 并加载
  libgazebo_ros_init/factory/force_system 插件）。
- 每台机器人的 gazebo_ros2_control 插件内部自带 controller_manager，控制器由 URDF 中
  <parameters> 指向的 YAML 配置（YAML 使用 /**/ 通配符匹配命名空间），并用
  controller_manager spawner 加载/激活。
- 机器人按顺序 spawn（避免多插件并发加载时共享 rcl 全局参数的竞态）。

用法：
    ros2 launch lunar_bringup lunar_bringup.launch.py gui:=false
"""
import os
import subprocess
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                             RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WS_TMP = os.path.join('/home', 'admina', 'MAS_ws', '.tmp')


def generate_launch_description():
    pkg_bringup = get_package_share_directory('lunar_bringup')
    pkg_desc = get_package_share_directory('lunar_descriptions')
    pkg_gazebo = get_package_share_directory('lunar_gazebo')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    scenario_file = os.path.join(pkg_bringup, 'config', 'scenario.yaml')
    with open(scenario_file, 'r', encoding='utf-8') as f:
        scenario = yaml.safe_load(f)

    world_file = os.path.join(pkg_gazebo, 'worlds', scenario['world']['file'])

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('gui', default_value='false'))
    ld.add_action(DeclareLaunchArgument('eval_mode', default_value='baseline',
                                        description="baseline=规则基线, policy=训练策略"))
    # 无头环境：强制 Mesa 软件渲染，避免 NVIDIA GLX 上下文创建失败
    ld.add_action(SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'mesa'))
    ld.add_action(SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1'))

    def build(context, *args, **kwargs):
        gui = LaunchConfiguration('gui').perform(context).lower() == 'true'
        actions = []

        # 让 launch 内节点能找到 pylib（gymnasium/torch）与训练/推理代码（lunar_rl/lunar_env）
        ws = '/home/admina/MAS_ws'
        for p in [os.path.join(ws, 'pylib'),
                  os.path.join(ws, 'src', 'lunar_rl'),
                  os.path.join(ws, 'src', 'lunar_env')]:
            os.environ['PYTHONPATH'] = p + os.pathsep + os.environ.get('PYTHONPATH', '')

        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')]),
            launch_arguments={
                'world': world_file,
                'gui': 'true' if gui else 'false',
                'verbose': 'true',
                'server_required': 'true',
                'extra_gazebo_args': '-s libgazebo_ros_state.so',
            }.items(),
        ))

        os.makedirs(WS_TMP, exist_ok=True)
        urdf_dir = tempfile.mkdtemp(prefix='urdf_', dir=WS_TMP)

        names = list(scenario['robots'].keys())
        robots = {}

        for name in names:
            cfg = scenario['robots'][name]
            ns = cfg['namespace']
            xacro_path = os.path.join(pkg_desc, 'urdf', cfg['xacro'])
            urdf_path = os.path.join(urdf_dir, name + '.urdf')

            proc = subprocess.run(
                ['xacro', xacro_path], capture_output=True, text=True,
                env=dict(os.environ))
            if proc.returncode != 0:
                raise RuntimeError(f'xacro failed for {name}: {proc.stderr}')
            urdf_str = proc.stdout
            with open(urdf_path, 'w', encoding='utf-8') as f:
                f.write(urdf_str)

            rsp = Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                namespace=ns,
                name=name + '_rsp',
                parameters=[{'robot_description': urdf_str, 'use_sim_time': True,
                             'frame_prefix': ns + '/'}],
                output='screen',
            )

            spawn = Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', cfg['entity'],
                    '-topic', '/' + ns + '/robot_description',
                    '-robot_namespace', ns,
                    '-x', str(cfg['x']),
                    '-y', str(cfg['y']),
                    '-z', str(cfg['z']),
                    '-Y', str(cfg['yaw']),
                ],
                output='screen',
            )

            spawner = Node(
                package='controller_manager',
                executable='spawner',
                namespace=ns,
                arguments=list(cfg['controllers']) + [
                    '--controller-manager', '/' + ns + '/controller_manager'],
                output='screen',
            )

            extras = []
            if 'extra_nodes' in cfg:
                for extra in cfg['extra_nodes']:
                    if extra == 'scout_gait':
                        extras.append(Node(
                            package='lunar_control',
                            executable='scout_gait',
                            namespace=ns,
                            output='screen',
                        ))

            robots[name] = {
                'rsp': rsp,
                'spawn': spawn,
                'spawner': spawner,
                'extras': extras,
            }
            actions.append(rsp)

        # 完全串行：spawn → spawner(加载控制器) → 下一台 spawn，避免共享 rcl 全局参数竞态
        for i, name in enumerate(names):
            spawn = robots[name]['spawn']
            spawner = robots[name]['spawner']
            extras = robots[name]['extras']

            if i == 0:
                actions.append(spawn)

            # spawn 完成 → 加载控制器
            actions.append(RegisterEventHandler(
                event_handler=OnProcessExit(target_action=spawn, on_exit=[spawner])))

            # spawner 完成 → 附加节点 + 下一台 spawn
            next_actions = list(extras)
            if i + 1 < len(names):
                next_actions.append(robots[names[i + 1]]['spawn'])
            if next_actions:
                actions.append(RegisterEventHandler(
                    event_handler=OnProcessExit(target_action=spawner, on_exit=next_actions)))

        actions.append(TimerAction(period=2.0, actions=[Node(
            package='lunar_task',
            executable='resource_manager',
            parameters=[{'scenario_file': scenario_file}],
            output='screen',
        )]))

        actions.append(TimerAction(period=3.0, actions=[Node(
            package='lunar_perception',
            executable='resource_markers',
            parameters=[{'use_sim_time': True}],
            output='screen',
        )]))

        # 运动自检（与 gzserver 同进程组，绕开沙箱跨进程 DDS 限制；完成后自动退出）
        actions.append(TimerAction(period=12.0, actions=[Node(
            package='lunar_env',
            executable='motion_self_test',
            output='screen',
        )]))

        # 评估节点（与 gzserver 同进程组；跑满 max_steps，约 120s 仿真）
        # baseline=规则基线 / policy=训练策略 / collect=收集 Gazebo 示范数据
        eval_mode = LaunchConfiguration('eval_mode').perform(context)
        if eval_mode == 'policy':
            ckpt_path = os.path.join(WS_TMP, 'stage2_gz_dagger2.pt')
            eval_node = Node(
                package='lunar_env',
                executable='run_policy',
                arguments=[scenario_file, ckpt_path],
                output='screen',
            )
        elif eval_mode == 'collect':
            data_path = os.path.join(WS_TMP, 'gazebo_bc_data.pkl')
            eval_node = Node(
                package='lunar_env',
                executable='collect_gazebo_data',
                arguments=[scenario_file, data_path, '4'],
                output='screen',
            )
        elif eval_mode == 'dagger':
            ckpt_path = os.path.join(WS_TMP, 'stage2_gz_dagger2.pt')
            data_path = os.path.join(WS_TMP, 'stage2_dagger_data2.pkl')
            eval_node = Node(
                package='lunar_env',
                executable='collect_dagger_gazebo',
                arguments=[scenario_file, ckpt_path, data_path, '2'],
                output='screen',
            )
        else:
            eval_node = Node(
                package='lunar_env',
                executable='run_baseline',
                arguments=[scenario_file],
                output='screen',
            )
        actions.append(TimerAction(period=30.0, actions=[eval_node]))

        return actions

    ld.add_action(OpaqueFunction(function=build))
    return ld
