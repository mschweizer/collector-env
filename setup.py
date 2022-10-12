from setuptools import setup, find_packages

setup(name='collector-env',
      version='0.1',
      description='A simple gridworld environment used to empirically evaluate the online learning performance of PbRL '
                  'agents.',
      url='https://github.com/mschweizer/collector-env',
      author='Marvin Schweizer',
      author_email='schweizer@kit.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'gym-minigrid==1.0.3',
          'gym==0.21',
          'matplotlib==3.5.1',
      ],
      include_package_data=True,
      python_requires='>=3.7',
      )
