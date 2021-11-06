from setuptools import setup, find_packages

setup(
    name='crypto_gym',
    version='1.0.0',
    packages=find_packages(),

    author='Ninja Mike',
    author_email='mike@cybertron.ninja',

    install_requires=[
        'pillow<=7.2.0',
        'gym>=0.12.5',
        'numpy>=1.16.4',
        'pandas>=0.24.2',
        'matplotlib>=3.1.1',
        'prettyprint>=0.1.5',
        'requests>=2.25.1',
        'PyYaml==5.4.1'
    ],

    package_data={
        'crypto_gym': ['datasets/data/*']
    }
)
