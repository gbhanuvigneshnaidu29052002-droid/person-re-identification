from setuptools import setup, find_packages

setup(
    name="person_re_identification",
    version="1.0.0",
    description="Deep Feature Embedding Pipeline for Multi-Camera Person Re-Identification",
    author="Bhanu Vignesh Naidu Ganeshna",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0"
    ],
    entry_points={
        'console_scripts': [
            'reid-eval=main:main',
        ],
    },
    python_requires='>=3.8',
)
