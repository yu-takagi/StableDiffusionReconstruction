'''
Setup script for bdpy

This file is a part of BdPy.
'''


from setuptools import setup


VERSION = '0.18'

if __name__ == '__main__':

    # Long description
    with open('./README.md') as f:
        long_description = f.read()

    # Setup
    setup(name='bdpy',
          version=VERSION,
          description='Brain decoder toolbox for Python',
          long_description=long_description,
          long_description_content_type='text/markdown',
          author='Shuntaro C. Aoki',
          author_email='brainliner-admin@atr.jp',
          maintainer='Shuntaro C. Aoki',
          maintainer_email='brainliner-admin@atr.jp',
          url='https://github.com/KamitaniLab/bdpy',
          license='MIT',
          keywords='neuroscience, neuroimaging, brain decoding, fmri, machine learning',
          packages=['bdpy',
                    'bdpy.bdata',
                    'bdpy.dataform',
                    'bdpy.distcomp',
                    'bdpy.dl',
                    'bdpy.dl.torch',
                    'bdpy.evals',
                    'bdpy.feature',
                    'bdpy.fig',
                    'bdpy.ml',
                    'bdpy.mri',
                    'bdpy.opendata',
                    'bdpy.preproc',
                    'bdpy.recon',
                    'bdpy.recon.torch',
                    'bdpy.stats',
                    'bdpy.util'],
          install_requires=[
              'numpy',
              'scipy',
              'scikit-learn',
              'h5py',
              'hdf5storage',
              'pyyaml'
          ])
