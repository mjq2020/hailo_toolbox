from setuptools import setup, find_packages


def get_version():
    with open("hailo_toolbox/__version__.py", "r") as f:
        version = f.read().strip()
    version = version.split("=")[1].strip().strip('"')
    return version


def get_install_requires():
    with open("requirements.txt", "r") as f:
        install_requires = f.read().splitlines()
    install_requires = [
        req.strip() for req in install_requires if not req.startswith("#")
    ]
    return install_requires


setup(
    name="hailo_toolbox",
    version=get_version(),
    packages=find_packages(),
    install_requires=get_install_requires(),
    entry_points={
        "console_scripts": [
            "hailo-toolbox=hailo_toolbox.cli.infer:main",
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
