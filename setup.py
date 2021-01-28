from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name="smg-detectron2",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="Wrapper for Detectron2",
    long_description="",  #long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-detectron2",
    packages=find_packages(include=["smg.detectron2", "smg.external.*"]),
    include_package_data=True,
    install_requires=[
        "detectron2",
        "numpy",
        "torch @ https://download.pytorch.org/whl/cu101/torch-1.7.0%2Bcu101-cp37-cp37m-win_amd64.whl",
        "torchaudio @ https://download.pytorch.org/whl/torchaudio-0.7.0-cp37-none-win_amd64.whl",
        "torchvision @ https://download.pytorch.org/whl/cu101/torchvision-0.8.1%2Bcu101-cp37-cp37m-win_amd64.whl"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
