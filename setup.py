import os
import setuptools
import sys


def compile_disort():
    repository_path = os.path.dirname(os.path.realpath(__file__))
    python = sys.executable
    module_name = 'disort'
    os.chdir(os.path.join(repository_path, 'disort4.0.98/'))
    os.system('{} -m numpy.f2py -c BDREF.f DISOBRDF.f DISORT.f ERRPACK.f LAPACK.f LINPAK.f RDI1MACH.f -m {}'.format(python, module_name))


def setup_pyRT_DISORT():
    setuptools.setup(
        name='pyRT_DISORT',
        version='0.0.1',
        author='kconnour',
        description='This does RT',
        packages=setuptools.find_packages(),
        include_package_data=True,
        python_requires='>=3.8',
        install_requires=[
            'numpy>=1.19.1'
        ],
        package_data={'': ['disort*.so']}
    )


compile_disort()
setup_pyRT_DISORT()
