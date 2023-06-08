import setuptools

setuptools.setup(
    name="disco",
    version="0.1.0",
    author="Yihao Chen",
    author_email="chenyihao@idea.edu.cn",
    description="DisCo-CLIP: A Distributed Contrastive Loss for Memory Efficient CLIP Training",
    url="https://github.com/IDEA-Research/DisCo-CLIP",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.7",
    ],
    extras_require={},
)
