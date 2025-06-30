import setuptools
from setuptools import find_packages

setuptools.setup(
    name='CGF',
    install_requires=[
          'numpy',
          'pandas',
          'git+https://github.com/PatriciaLucas/AutoML.git',
          'git+https://github.com/PYFTS/pyFTS',
          'torch'
          ],
    packages=find_packages(),
    version='1.0',
    description='CGF',
    long_description='CAUSAL GRAPH FUZZY',
    long_description_content_type="text/markdown",
    author='Patrícia de Oliveira e Lucas',
    author_email='patricia.lucas@ifnmg.edu.com',
    url='',
    download_url='',
    keywords=['time series', 'causal graph', 'llm', 'fuzzy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
    
    ]
)
