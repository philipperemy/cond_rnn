from setuptools import setup

setup(
    name='cond-rnn',
    version='1.0',
    description='Conditional RNN',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['cond_rnn'],
    install_requires=[
        'tensorflow>=1.13.1'
    ]
)
