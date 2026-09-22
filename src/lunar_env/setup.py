from setuptools import find_packages, setup

package_name = 'lunar_env'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='admina',
    maintainer_email='admina@example.com',
    description='Gymnasium training interface wrapping the ROS2/Gazebo lunar simulation.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'random_policy = lunar_env.random_policy:main',
            'motion_self_test = lunar_env.motion_self_test:main',
            'turn_test = lunar_env.turn_test:main',
            'run_baseline = lunar_env.run_baseline:main',
            'run_policy = lunar_env.run_policy:main',
            'collect_gazebo_data = lunar_env.collect_gazebo_data:main',
            'collect_dagger_gazebo = lunar_env.collect_dagger_gazebo:main',
        ],
    },
)
