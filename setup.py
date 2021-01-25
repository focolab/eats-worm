import setuptools

setuptools.setup(
    name="gcamp_extractor",
    version="0.0.3",
    author="Steven Ban",
    author_email="ban.steven1337@gmail.com",
    description="Method for extracting GCaMP signal from volumetric imaging recordings",
    long_description_content_type=open('README.md').read(),
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'numpy>=1.13.3',
          'scipy>=1.0.0',
          'tifffile>=0.15.1',
          'opencv-python>=4.1.0.25',
          'matplotlib>=2.1.0',
          'imreg_dft',
          'fastcluster',
          'napari[all]',
          'pyqtgraph',
          'magicgui',
          'scikit-image'
      ], 
    dependency_links=['https://github.com/matejak/imreg_dft/tarball/master#egg=imreg_dft'],
    python_requires='>=3.6',
)
