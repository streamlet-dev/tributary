import sys
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if sys.version_info.major < 3:
    with open(path.join(here, 'requirements_py2.txt'), encoding='utf-8') as f:
        requires = f.read().split()
else:
    with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
        requires = f.read().split()


setup(
    name='tributary',
    version='0.0.6',
    description='Analytics library',
    long_description=long_description,
    url='https://github.com/timkpaine/tributary',
    download_url='https://github.com/timkpaine/tributary/archive/v0.0.6.tar.gz',
    author='Tim Paine',
    author_email='timothy.k.paine@gmail.com',
    license='Apache 2.0',
    install_requires=requires,
    extras_require={'dev': requires + ['pytest', 'pytest-cov', 'pylint', 'flake8']},

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='analytics tools plotting',

    packages=find_packages(exclude=['tests', ]),
    package_data={},
    include_package_data=True,
    zip_safe=False,
    entry_points={
    }
)
