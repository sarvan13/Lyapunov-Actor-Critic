from setuptools import setup

setup(name='cart_pole',
      version='0.0.1',
      install_requires=["gymnasium",
                        "pygame>=2.1.3"],
      packages=["env", "env.cart_pole"],)