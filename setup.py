from setuptools import setup

setup(
    name='cond-rnn',
    version='2.1',
    description='Conditional RNN',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['cond_rnn'],
    install_requires=[
        'numpy',
        'tensorflow>=2.x',
    ]
)
