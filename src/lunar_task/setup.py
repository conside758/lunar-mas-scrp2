from setuptools import find_packages, setup

package_name = 'lunar_task'

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
    description='Task allocation / scheduling layer and resource manager.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'resource_manager = lunar_task.resource_manager:main',
        ],
    },
)
