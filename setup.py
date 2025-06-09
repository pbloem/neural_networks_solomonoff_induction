from setuptools import setup

setup(name='nnsi',
      version='0.1',
      description='NNSI',
      url='peterbloem.nl',
      author='Peter Bloem',
      author_email='up@peterbloem.nl',
      license='MIT',
      packages=['nnsi'],
      install_requires=[
            'absl-py',
            'dm-haiku',
            'dm-tree',
            'jax',
            'jaxtyping',
            'numpy',
            'tqdm',
            'dm-haiku',
      ],
      zip_safe=False)