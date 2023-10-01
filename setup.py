from setuptools import find_packages, setup

# Package meta-data
NAME = "safe_rl_algos"
DESCRIPTION = "Safe reinforcement learning algorithms"
URL = "https://github.com/dobro12/Safe-RL-Algos"
EMAIL = "rlaehgudhoho@gmail.com"
AUTHOR = "Dohyeng Kim"
VERSION = "1.0.0"

# What packages are required for this module to be executed?
REQUIRED = [
    "gymnasium",
    "safety-gymnasium",
    "torch",
    "qpsolvers==1.9.0",
    "scipy",
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    long_description=long_description,
    long_description_content_type = "text/markdown",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=REQUIRED,
    python_requires='>=3.8',
    )