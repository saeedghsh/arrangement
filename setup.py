#!/usr/bin/env python

from distutils.core import setup
import codecs
import os

# Get the long description from the README file
here = os.path.abspath(path.dirname(__file__))
with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='arrangement',
      version='0.4', # major.minor[.patch[.sub]].
      description='2D Arrangement in Python',
      long_description=long_description,
      author='Saeed Gholami Shahbandi',
      author_email='saeed.gh.sh@gmail.com',
      maintainer='Saeed Gholami Shahbandi',
      maintainer_email='saeed.gh.sh@gmail.com',
      url='https://github.com/saeedghsh/arrangement',
      packages=['arrangement',],
      keywords='2D arrangement computational geometry',
      license='BSD'
     )
