from setuptools import find_packages, setup

with open("autoforce/version.py") as f:
    _version: dict[str, str] = {}
    exec(f.read(), _version)
    __version__ = _version["__version__"]


setup(
    name="autoforce",
    version=__version__,
    author="Amir Hajibabaei",
    author_email="autoforcefield@gmail.com",
    description="machine learning force-fields",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=["torch", "ase", "numpy", "scipy"],
    url="https://github.com/AutoForceField/AutoForce",
    license="MIT",
)
