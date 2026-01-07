# setup.py
from setuptools import setup, find_packages

setup(
    name="semantic_advination_core",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[],
    author="Semantic Advination Team",
    author_email="support@semantic-advination.com",
    description="Система семантического предсказания команд",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)