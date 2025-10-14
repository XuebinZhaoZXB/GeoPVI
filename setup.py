from setuptools import setup, find_packages

# Setting up
setup(
      name = 'geopvi',
      version = '1.0.0',
      author = 'Xuebin Zhao',
      author_email = '<xuebin.zhao@ed.ac.uk>',
      description = 'Geophysical inversion using parametric variational inference',
      packages = find_packages(),      
      keywords=['variational inference', 'geophysical inversion']
      )
