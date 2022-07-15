from setuptools import setup, find_packages

setup(name = 'yaket',
      version = '0.0.1',
      description = 'YAml KEras Trainer for quick AI development',
      author = 'Andrea Favia',
      author_email = 'andrea.favia@pm.me',
      url = '',
      packages = find_packages(include = ['yaket', 'yaket.*']),
      setup_requires = ['flake8'],
      )
