import setuptools
from numba import cuda

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

required_packages = ['numpy', 'einops']
try:
    import cupy
except ModuleNotFoundError:
    cupy = 'cupy-cuda101'
    if cuda.is_available():
        cupy = 'cupy-cuda'+''.join(map(str, cuda.runtime.get_version()))
    required_packages.append(cupy)

setuptools.setup(
    name="flymodel",
    version="0.0.1",
    author="Patrick Ramos, Ryan Ramos",
    author_email="patrickjcor@gmail.com, ryanccor@gmail.com",
    description="Unofficial implementation of \"Algorithmic insights on continual learning from fruit flies\"",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ramos-Ramos/flymodel",
    project_urls={
        "Bug Tracker": "https://github.com/Ramos-Ramos/flymodel/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=required_packages
)