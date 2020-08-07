import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension


requirements = ["torch","transformers"]

setup(
    name="ViNLP",
    version="1.0",
    author="Nguyen Van Nha",
    url="https://github.com/bino282/ViNLP.git",
    description="A NLP toolkit for Vietnamese",
    install_requires=requirements,
    packages=find_packages(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
