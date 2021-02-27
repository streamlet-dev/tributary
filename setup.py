import io
import os.path
from codecs import open

from setuptools import find_packages, setup

pjoin = os.path.join
here = os.path.abspath(os.path.dirname(__file__))
name = "tributary"


def get_version(file, name="__version__"):
    path = os.path.realpath(file)
    version_ns = {}
    with io.open(path, encoding="utf8") as f:
        exec(f.read(), {}, version_ns)
    return version_ns[name]


version = get_version(pjoin(here, name, "_version.py"))

with open(pjoin(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read().replace("\r\n", "\n")


requires = [
    "aioconsole>=0.2.1",
    "aiofiles>=0.4.0",
    "aiohttp>=3.5.4",
    "aiohttp-sse>=2.0",
    "aiohttp-sse-client>=0.2.0",
    "aiokafka>=0.6.0",
    "aiostream>=0.3.1",
    "asyncpg>=0.20.1",
    "beautifulsoup4>=4.9.1",
    "boltons>=20.1.0",
    "emails>=0.5.15",
    "future>=0.17.1",
    "gevent>=1.3.7",
    "graphviz>=0.10.1",
    "ipython>=7.0.1",
    "ipydagred3>=0.1.5",
    "numpy>=1.15.3",
    "pandas>=0.19.0",
    "scipy>1.2.0",
    "six>=1.11.0",
    "socketIO-client-nexus>=0.7.6",
    "sympy>=1.5.1",
    "temporal-cache>=0.0.6",
    "tornado>=5.1.1",
    "twilio>=6.50.1",
]

requires_dev = [
    "black>=20.",
    "flake8>=3.7.8",
    "flake8-black>=0.2.1",
    "mock",
    "pybind11>=2.4.0",
    "pytest>=4.3.0",
    "pytest-cov>=2.6.1",
    "Sphinx>=1.8.4",
    "sphinx-markdown-builder>=0.5.2",
] + requires

setup(
    name=name,
    version=version,
    description="Analytics library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/timkpaine/{name}".format(name=name),
    author="Tim Paine",
    author_email="timothy.k.paine@gmail.com",
    license="Apache 2.0",
    install_requires=requires,
    extras_require={
        "dev": requires_dev,
        "functional": ["confluent-kafka>=0.11.6", "websocket_client>=0.57.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="analytics tools plotting",
    packages=find_packages(exclude=["tests"]),
    package_data={},
    include_package_data=True,
    zip_safe=False,
    entry_points={},
)
