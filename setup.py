from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='LLMs',
    version='1.0',
    author='ywchoi',
    author_email='chldyddnjs@naver.com.com',
    #long_description=read('README.md'),
    python_requires='>=3.11',
    #install_requires=['numpy'],
    #package_data={'mypkg': ['*/requirements.txt']},
    #dependency_links = [], ## 최신 패키지를 설치하는 경우 
    description='My LLMs project',
    packages=find_packages(include=['LLMs'])
)

