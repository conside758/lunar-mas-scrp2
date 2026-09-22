from setuptools import find_packages, setup

package_name = 'lunar_rl'

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
    description='Fast surrogate environment and MARL algorithms for the lunar mining task.',
    license='Apache-2.0',
)
