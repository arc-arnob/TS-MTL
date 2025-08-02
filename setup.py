# setup.py
from setuptools import setup, find_packages

setup(
    name="TS_MTL",
    version="0.1.0",
    description="Your time-series multi-task learning package",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "python-dateutil", 

    ],
    entry_points={
        "console_scripts": [
            "ts-mtl=TS_MTL.cli:main",
        ],
    },
)
