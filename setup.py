import setuptools
import versioneer


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    author="Gavenski, Nathan",
    name='tensorboard-wrapper',
    cmdclass=versioneer.get_cmdclass(),
    version=versioneer.get_version(),
    author_email="nathan.gavenski@edu.pucrs.br",
    description="A wrapper class for the Tensorflow's Tensorboard application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NathanGavenski/Tensorboard",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
 )
