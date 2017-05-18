from setuptools import setup, find_packages

setup(
    name='hse',
    version='0.1dev',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "tensorflow",
    ],
)
