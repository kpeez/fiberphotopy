from setuptools import setup, find_packages

setup(
    name="fiberphotopy",
    version="0.0.0",
    description="Package for analyzing fiber photometry data",
    author="Kyle P",
    author_email="krpuhger@gmail.com",
    packages=find_packages(where="fiberphotopy"),
    package_dir={"": "fiberphotopy"},
    # packages=find_packages(include=["fiberphotopy", "fiberphotopy.*"]),
    scripts={},
    install_requires=["pandas", "numpy", "matplotlib", "seaborn"],
)
