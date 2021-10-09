"""Setup file"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()

classifiers = [
  "Programming Language :: Python :: 3",
  ("License :: OSI Approved :: "
   "GNU Affero General Public License v3 or later (AGPLv3+)"),
  "Operating System :: Unix"]

setup_kwargs = {
  "name": "popto8bit",
  "version": "0.0.1",
  "author": "Shih-Yang Su",
  "author_email": "at7788546@gmail.com",
  "description": "Automatic conversion of pop music into chiptunes",
  "long_description": long_description,
  "long_description_content_type": "text/markdown",
  "url": "https://github.com/LemonATsu/pop-to-8bit",
  "packages": find_packages(),
  "include_package_data": True,
  "entry_points": {'console_scripts': ['popto8bit = popto8bit:main']},
  "install_requires": ['librosa',
                       'pypropack',
                       'vamp'],
  "classifiers": classifiers
}

setup(**setup_kwargs)
