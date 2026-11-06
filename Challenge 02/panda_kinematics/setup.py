from setuptools import find_packages, setup

package_name = 'panda_kinematics'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Team 5',
    maintainer_email='Mahmoud.Ghobashy@student.guc.edu.eg',
    description='Panda FK/IK (Challenge 1), velocity/acceleration kinematics and trajectory planning (Challenge 2)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kinematics = panda_kinematics.kinematics:main',
            'trajectory_node = panda_kinematics.trajectory_node:main',
        ],
    },
)
