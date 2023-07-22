from setuptools import setup, find_packages

from odm import __version__

setup(
    name="ODMammogram",
    version=__version__,
    url="https://github.com/RyanZurrin/ODMammogram",
    author="Ryan Zurrin, Daniel Haehn",
    author_email="ryan.zurrin001@umb.edu, daniel.haehn@umb.edu",
    description="Outlier Detection for Mammograms (ODM) is a Python package for medical "
    "image analysis with a focus on detection of outliers in DICOM mammogram "
    "images. Leveraging techniques such as 5-bin histogram thresholding and "
    "variational auto-encoders, ODM facilitates automated identification of "
    "images with atypical characteristics. Its user-friendly API allows seamless "
    "integration with existing pipelines, making it a handy tool for large-scale "
    "studies and quality assurance in mammography medical imaging research.",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    install_requires=[
        "matplotlib~=3.6.2",
        "mahotas~=1.4.13",
        "pyod",
        "scikit-learn~=1.2.0",
        "scikit-image==0.19.2",
        "keras",
        "hyperopt~=0.2.7",
        "pydicom~=2.3.1",
        "gdcm",
        "pylibjpeg[all]",
        "numpy~=1.23.5",
        "tqdm~=4.65.0",
        "Pillow~=9.3.0",
        "pandas~=1.5.2",
        "statsmodels~=0.13.5",
        "tensorflow~=2.12.0",
        "future~=0.18.3",
        "setuptools~=65.6.3",
    ],
)
