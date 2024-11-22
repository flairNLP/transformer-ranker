from setuptools import setup, find_packages


def read_requirements():
    with open('requirements.txt') as req:
        return req.read().strip().split('\n')


setup(
    name='transformer-ranker',
    version='0.1.1',
    packages=find_packages(),
    description='Efficiently find the best-suited language model (LM) for your NLP task',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author='Lukas Garbas',
    author_email='lukasgarba@gmail.com',
    url="https://github.com/flairNLP/transformer-ranker",
    install_requires=read_requirements(),
    license='MIT',
    python_requires=">=3.9",
)
