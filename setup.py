import atexit
import glob
import os
import setuptools
import sys


def compile_disort():
    os.chdir(os.path.join(package_path, 'disort4.0.98/'))
    os.system('{} -m numpy.f2py -c BDREF.f DISOBRDF.f DISORT.f ERRPACK.f LAPACK.f LINPAK.f RDI1MACH.f -m {}'.format(
        sys.executable, 'disort'))


def get_library_name():
    os.chdir(os.path.join(package_path, 'disort4.0.98/'))
    disort_binary = glob.glob('*.so')[0]
    return disort_binary


def move_library(lib_name):
    os.rename(os.path.join(os.path.join(package_path, 'disort4.0.98/'), lib_name),
              os.path.join(package_path, lib_name))


def setup_package(lib_name):
    os.chdir(package_path)
    setuptools.setup(
        name='pyRT_DISORT',
        version='0.0.1',
        description='Make radiative transfer more accessible to the yearning masses',
        url='https://github.com/kconnour/pyRT_DISORT',
        author='kconnour',
        packages=setuptools.find_packages(),
        include_package_data=True,
        python_requires='>=3.8',
        install_requires=[
            'numpy>=1.19.1',
            'pdoc3>=0.9.1'
        ],
        # I have no clue why I need to go up a directory when I'm already in the correct directory
        package_data={'': ['../{}'.format(lib_name)]}
    )


# Define paths specific to the user
package_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(package_path, 'pyRT_DISORT')

# Do the installation
compile_disort()
disort = get_library_name()
move_library(disort)
setup_package(disort)
