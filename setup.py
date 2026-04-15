"""
setup.py
--------
Makes the `src` package importable during development.

Install in editable mode with:
    pip install -e .
"""

from setuptools import find_packages, setup

setup(
    name="instruct-tune-llm",
    version="1.0.0",
    description="Domain-Specific Instruction Tuning of Large Language Models via LoRA",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Harshit",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "peft>=0.9.0",
        "bitsandbytes>=0.43.0",
        "pyyaml>=6.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
    ],
    extras_require={
        "tracking": ["wandb>=0.16.0"],
        "dev":      ["black>=24.0.0", "isort>=5.13.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
