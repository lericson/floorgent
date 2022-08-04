from pathlib import Path
from setuptools import setup

requirements_txt = Path(__file__).parent / 'requirements.txt'
deps = requirements_txt.read_text().splitlines()

setup(name='floorgent',
      version='0.1',
      description='FloorGenT',
      url='https://github.com/lericson/floorgent',
      author='Ludvig Ericson',
      author_email='github@lericson.se',
      packages=['floorgent'],
      install_requires=deps,
      license='Apache 2.0')
