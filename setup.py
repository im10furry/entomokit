from setuptools import setup, find_packages

setup(
    name="insect-synthesizer",
    version="0.1.0",
    description="A Python tool for segmenting insects using multiple methods (SAM3, Otsu, GrabCut) and synthesizing images",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/insect-synthesizer",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "segmentation": [
            "torch>=2.0.0,<2.4.0",
            "torchvision>=0.15.0,<0.19.0",
            "opencv-python>=4.8.0",
        ],
        "cleaning": [
            "imagehash",
        ],
        "video": [
            "opencv-python>=4.8.0",
        ],
        "data": [
            "pandas",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "insect-segment=scripts.segment:main",
            "insect-clean=scripts.clean_figs:main",
            "insect-extract=scripts.extract_frames:main",
            "insect-split=scripts.split_dataset:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
