"""Setup script for HalluGuard package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="halluguard",
    version="0.1.0",
    author="Arnab Mondal",
    description=(
        "Real-Time Hallucination Detection and Mitigation for RAG Systems"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arnabmondal/halluguard",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "halluguard-benchmark=evaluation.benchmark:main",
        ],
    },
)
