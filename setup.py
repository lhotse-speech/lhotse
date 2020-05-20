# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='lhotse',
    version='0.1',
    description='Package for managing speech data in K2.',
    author='Piotr Żelasko, Jan Trmal, Daniel Povey',
    packages=find_packages(),
    # The line below makes every script in the list an executable that's inserted in PATH
    # as long as the virtualenv/conda env is active; they can be used like any other shell program
    scripts=['lhotse/bin/lhotse']
)
