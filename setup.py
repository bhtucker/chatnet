"""
Chatnet
~~~~~~~

Classication learning on chat data
"""

from setuptools import setup, find_packages


def get_requirements(suffix=''):
    with open('requirements%s.txt' % suffix) as f:
        result = f.read().splitlines()
    return result


def get_long_description():
    with open('readme.md') as f:
        result = f.read()
    return result

setup(
    name='Chatnet',
    version='0.1.0',
    url='https://github.com/bhtucker/chatnet',
    author='Benson Tucker',
    author_email='bensontucker@gmail.com',
    description='Classication learning on chat data',
    long_description=get_long_description(),
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    platforms='any'
)
