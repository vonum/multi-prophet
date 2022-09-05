import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multi-prophet",
    version="1.1.1",
    author="Milan Keca",
    author_email="vonum.mk@gmail.com",
    description="Multivariate forecasting using Facebook Prophet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vonum/multi-prophet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "prophet=1.0",
        "pandas",
        "plotly"
    ]
)
