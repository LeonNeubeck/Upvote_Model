from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='Upvote_Model',
      version="1.0",
      description="Project Description",
      packages=['app','models','raw_data','scripts','Upvote_Model'],
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/Upvote_Model-run'],
      zip_safe=False)
