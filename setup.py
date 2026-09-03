from setuptools import find_packages, setup

package_name = 'sol_slam'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kalvik Jakkala',
    maintainer_email='itskalvik@gmail.com',
    description='Sonar-Only Local SLAM',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'sonoptix_echo = sol_slam.sonoptix_echo:main',
        ],
    },
)
