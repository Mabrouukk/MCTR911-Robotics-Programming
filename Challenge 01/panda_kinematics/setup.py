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
    description='Forward and inverse position kinematics for the Franka Emika Panda',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kinematics = panda_kinematics.kinematics:main',
        ],
    },
)
