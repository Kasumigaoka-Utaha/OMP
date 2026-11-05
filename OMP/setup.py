from setuptools import find_namespace_packages, setup
from pathlib import Path

ROOT = Path(__file__).resolve().parent

setup(
    name='omp-policy',
    version='0.1.0',
    description='OMP: One-step Meanflow Policy with Directional Alignment',
    long_description=(ROOT.parent / 'README.md').read_text(encoding='utf-8')
        if (ROOT.parent / 'README.md').exists() else '',
    long_description_content_type='text/markdown',
    author='Han Fang, Yize Huang, Yuheng Zhao, Paul Weng, Xiao Li, Yutong Ban',
    license='MIT',
    python_requires='>=3.8',
    packages=find_namespace_packages(include=['omp', 'omp.*'],
                                     exclude=['*.__pycache__', '*.__pycache__.*']),
    package_data={
        'omp': ['config/*.yaml', 'config/task/*.yaml'],
    },
    include_package_data=True,
    project_urls={
        'Paper': 'https://proceedings.mlr.press/v306/fang26h.html',
    },
)
