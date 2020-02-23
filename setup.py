import os
from setuptools import setup, find_packages, Command


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import subprocess #import needed for programmatic calls with distutils
        subprocess.Popen('rm -rf ./build ./*.pyc ./*.egg-info', shell=True).wait()


requirements = ''
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt.txt')) as f:
    requirements = f.readlines()

setup(
    name='linguini',
    description='NLP framework built on top of TensorFlow 2',
    author='Dionysis Varelas',
    author_email='dionvarelas@gmail.com',
    version='0.1.0',
    license='proprietary',
    url='',
    install_requires=requirements,
    platforms='any',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    cmdclass={
        'clean': CleanCommand,
    },
    packages=find_packages(include=['linguini*']),
)