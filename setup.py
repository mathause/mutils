from setuptools import setup, find_packages

setup(name='mutils',
      version='0.2.0',
      description="Mathias' utils",
      url='',
      author='Mathias Hauser',
      author_email='mathias.hauser@env.ethz.ch',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy',
                        ],
      zip_safe=False)
