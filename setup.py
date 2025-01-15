from setuptools import setup, find_packages

setup(
    name='custom_envs',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'mujoco',
        # Add other dependencies here
    ],
)