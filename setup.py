import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="landmark",
    version="0.0.1",
    author="Jake Searcy",
    author_email="jsearcy@uoregon.edu",
    description="A package for automatically landmarking 3d body scans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jsearcy/landmark",
    project_urls={
        "Bug Tracker": "https://github.com/jsearcy/landmark",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
