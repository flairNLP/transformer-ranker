from setuptools import setup, find_packages


def read_requirements():
    with open('requirements.txt') as req:
        return req.read().strip().split('\n')


setup(
    name='transformer-ranker',
    version='0.1.0',
    packages=find_packages(),
    description='Rank transformer models for NLP tasks using transferability measures',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author='Lukas Garbas',
    author_email='lukasgarba@gmail.com',
    url="https://github.com/flairNLP/transformer-ranker",
    install_requires=read_requirements(),
    python_requires=">=3.8",
)
