from setuptools import setup, find_packages

setup(
    name="jump_tank",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Custom library for Jump Tank project",
    # long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/QR-Notes",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
