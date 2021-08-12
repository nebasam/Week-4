#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['torch','torchaudio','IPython', 'matplotlib', 'librosa', 
'numpy', 'pandas','numpy', 'pandas']

test_requirements = ['pytest>=3', ]

setup(
    author="Nebiyu Samuel",
    email="neba.samuel17@gmail.com",
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Speech to text for Swahili Language",
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='modules,data preprocess, data augmentation , feature extraction',
    name='pharmaceutical-sales-Forecasting',
    packages=find_packages(include=['modules']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/10Academy-Group-4/Week-4/tree/Data_preprocess_neba',
    version='0.1.0',
    zip_safe=False,
)
