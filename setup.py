from setuptools import setup
from setuptools import find_packages


setup(name='KerasText',
      version='0.1',
      description='Text Processing Layers for Keras',
      author='Sunil Mohan',
      author_email='sunilm.k3@gmail.com',
      url='https://github.com/',
      download_url='https://github.com/',
      license='MIT',
      install_requires=['keras>=2.0.0',
                        ],
      extras_require={
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
          # 'Development Status :: 4 - Beta',
          # 'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
