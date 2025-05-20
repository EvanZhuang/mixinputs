
from setuptools import setup, find_packages

def read_readme():
    with open('README.md') as f:
        return f.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'vllm>=0.8.5',
]

setup(
    name='mixinputs',
    version='0.1.0',
    description='Mixture of Inputs Patch for vLLM',
    long_description= read_readme(),
    long_description_content_type="text/markdown",
    author='Yufan Zhuang',
    author_email='y5zhuang@ucsd.edu',
    url='https://github.com/EvanZhuang/mixinputs',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=requirements,
    license='Apache License 2.0',
    entry_points={
        'console_scripts': ['mixinputs=mixinputs.commands:run'],
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
    ]
)