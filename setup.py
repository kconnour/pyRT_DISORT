import glob
import os
import setuptools
import sys


def compile_disort():
    repository_path = os.path.dirname(os.path.realpath(__file__))
    python = sys.executable
    module_name = 'disort'
    os.chdir(os.path.join(repository_path, 'disort4.0.98/'))
    os.system('{} -m numpy.f2py -c BDREF.f DISOBRDF.f DISORT.f ERRPACK.f LAPACK.f LINPAK.f RDI1MACH.f -m {}'.format(python, module_name))
    file = glob.glob('*.so')[0]
    os.rename('{}{}'.format(os.path.join(repository_path, 'disort4.0.98/'), file), '{}/{}'.format(repository_path, file))
    os.chdir(repository_path)
    return file


def setup_pyRT_DISORT(file):
    setuptools.setup(
        name='pyRT_DISORT',
        version='0.0.1',
        author='kconnour',
        description='This does RT',
        packages=setuptools.find_packages(),
        #include_package_data=False,
        include_package_data=True,
        python_requires='>=3.8',
        install_requires=[
            'numpy>=1.19.1'
        ],
        package_data={'': ['../{}'.format(file)]}
    )


file = compile_disort()
setup_pyRT_DISORT(file)
