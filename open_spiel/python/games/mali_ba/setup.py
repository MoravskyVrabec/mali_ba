from setuptools import setup, find_packages

setup(
    name="mali-ba",
    version="0.3",
    packages=find_packages(),
    package_data={
        "mali_ba": ["*.py"],  # Include all Python files at the root level
    },
    install_requires=[
        "pygame",
    ],
    description="Mali-Ba board game implementation using OpenSpiel",
    author="Rob Parker",
    python_requires=">=3.7",
)