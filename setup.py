from setuptools import setup, find_packages

setup(
    name="genetic_algorithm_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "deap",
        "numpy",
        "matplotlib",
        "scikit-learn"
    ],
)