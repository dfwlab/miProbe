from setuptools import setup, find_packages

setup(
    name='miprobe',
    version='0.1.0',
    description='miProbe Toolkit: Microbial peptide embedding and AI-ready interface',
    author='Dingfeng Wu',
    author_email='dfw_bioinfo@126.com',
    packages=find_packages(include=["miprobe", "miprobe.*"]),
    install_requires=['requests', 'numpy', 'torch'],
    python_requires='>=3.8',
)
