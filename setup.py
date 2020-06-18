import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DRT_Lib",
    version="0.0.1",
    author="Giang Le",
    author_email="giangtle@uw.edu",
    description="Python implementation of different algorithms for calculating electrochemical impedance distribution of relaxation times",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giangtle/DRT_Lib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
