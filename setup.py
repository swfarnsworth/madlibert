from setuptools import setup, find_packages


setup(
    name='madlibert',
    packages=find_packages(),
    version='0.0.0',
    description='Fill in the blank with BERT',
    author='Steele Farnsworth',
    install_requires=[
        'torch',
        'transformers'
    ]
)
