from setuptools import setup
import sys

if sys.version_info[:2] != (3, 8):
    raise RuntimeError("Python version 3.8 required")

setup(
    name='WFlib',
    version='0.1',
    description='Library for website fingerprinting attacks',
    author='Xinhao Deng',
    packages=[
        "WFlib",
        "WFlib.models",
        "WFlib.tools"
    ],
    install_requires=[
        "tqdm",
        "numpy",
        "pandas",
        "scikit-learn",
        "einops",
        "timm",
        "pytorch-metric-learning",
        "captum"
    ],
)