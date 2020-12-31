from setuptools import setup

with open("README.md", encoding="utf-8") as f:
      long_description = f.read().strip()

setup(name='yft',
      version='0.1.0',
      description='Utility functions for text processing',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://github.com/liao961120/yft',
      author='Yongfu Liao',
      author_email='liao961120@github.com',
      license='MIT',
      packages=['yft'],
      install_requires=['tqdm', 'networkx'],
      #tests_require=['cqls'],
      zip_safe=False)
