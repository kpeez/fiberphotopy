from setuptools import setup, find_packages
import pathlib

setup(
    here=pathlib.Path(__file__).parent.resolve(),
    name="fiberphotopy",
    version="0.0.1",
    description="Package for analyzing fiber photometry data",
    author="Kyle P",
    author_email="krpuhger@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
