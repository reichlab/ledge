from setuptools import find_packages, setup
from os import path

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open(path.join("ledge", "__version__.py")) as fp:
    version = fp.read().split("=")[1].strip()[1:-1]

setup(
    name="ledge",
    version=version,
    description="Hedging algorithms with lag",
    long_description=readme,
    author="Abhinav Tushar",
    author_email="lepisma@fastmail.com",
    url="https://github.com/reichlab/ledge",
    install_requires=[],
    keywords="",
    license="GPLv3",
    packages=find_packages(),
    classifiers=(
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only"
    ))
