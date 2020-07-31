import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='tensorboard-wrapper',  
    version='0.1',
    author="Gavenski, Nathan",
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
