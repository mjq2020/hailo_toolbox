from setuptools import setup, find_packages

setup(
    name="hailo_toolbox",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "onnx",
        "onnxruntime",
        "opencv-python",
        "pillow",
        "pyyaml",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "dl-convert=hailo_toolbox.cli.convert:main",
            "dl-infer=hailo_toolbox.cli.infer:main",
            "dl-server=hailo_toolbox.cli.server:main",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolbox for deep learning model conversion and inference",
    keywords="deep learning, inference, model conversion",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
