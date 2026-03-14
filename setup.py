from setuptools import setup, find_packages

setup(
    name="stationaritytoolkit",
    version="2.0.3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
