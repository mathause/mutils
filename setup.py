from setuptools import setup, find_packages

# get version
with open("mutils/version.py") as f:
    l = f.readline().strip().replace(' ', '').replace('"', '')
    version = l.split('=')[1]
__version__ = version

setup(name='mutils',
      version=__version__,
      description="Mathias' utils",
      url='',
      author='Mathias Hauser',
      author_email='mathias.hauser@env.ethz.ch',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'pandas', 'matplotlib'
                        ],
      zip_safe=False)
