import setuptools
from setuptools import setup
from os import path

ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

setup(
    name="metriX",
    version="0.1",
    description="A library containing a collection of distance and similarity measures to compare time series data.",
    author="Kay Hansel, Theo Gruner, Firas Al-Hafez ",
    author_email="kay.hansel@tu-darmstadt.de,  theo_sunao.gruner@tu-darmstadt.de, firas.al-hafez@tu-darmstadt.de",
    packages=setuptools.find_packages(),
    install_requires=requires_list,
)