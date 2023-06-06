#import setuptools
from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="psie", # Replace with your own username
    version="0.0.9",
    author="Alex H. Parker",
    author_email="aparker@seti.org",
    description="The Planetary Science Inference Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alex-parker/psie",
    project_urls={
        "Bug Tracker": "https://github.com/alex-parker/psie/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

      package_dir={'psie': 'src'},
      packages=['psie'],

    python_requires=">=3.6",
)