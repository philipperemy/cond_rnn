import os
import platform

from setuptools import setup

tensorflow = 'tensorflow'
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    tensorflow = 'tensorflow-macos'
    # https://github.com/grpc/grpc/issues/25082
    os.environ['GRPC_PYTHON_BUILD_SYSTEM_OPENSSL'] = '1'
    os.environ['GRPC_PYTHON_BUILD_SYSTEM_ZLIB'] = '1'

install_requires = ['numpy', tensorflow, 'tensorflow_addons', "protobuf<=3.20"]

setup(
    name='cond-rnn',
    version='3.2.1',
    description='Conditional RNN',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['cond_rnn'],
    install_requires=install_requires
)
