from encodings import utf_8
from platform import python_version
from unicodedata import name
from setuptools import setup, find_packages

with open('README.md', 'r', encoding=utf_8) as file:
    readme = file.read()

setup(
    name='deeplearnprac',
    version='0.0.0',
    long_description=readme,
    author="MinhyukWoo",
    author_email="minhyukwoo.dev@gmail.com",
)