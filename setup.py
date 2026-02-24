from setuptools import setup, find_packages

setup(
    name="model-inference-server",
    version="0.1.0",
    author="Prabhath Chellingi",
    author_email="prabhathchellingi2003@gmail.com",
    description="A server for model inference using various machine learning models.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "Flask",
        "transformers",
        "sentence-transformers",
        "openai",
        "torch",
        "huggingface-hub",
        "numpy",
        "pynvml",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
