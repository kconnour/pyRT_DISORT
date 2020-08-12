import os
import pkg_resources
import sys


def check_required_libraries():
    required = 'numpy'
    installed = [pkg.key for pkg in pkg_resources.working_set]
    return required in installed


if __name__ == '__main__':
    install_directory = '/home/kyle/disTest'

    # 1. check that this python has numpy
    if not check_required_libraries():
        print('numpy is required to create pyRT_DISORT. Please install it before continuing')
        raise SystemExit(1)
    else:
        print('Prerequisite libraries are satisfied.')

    # 2. untar the tarball
    os.chdir(install_directory)
    os.system('tar -xf pyRT_DISORT.tar.gz')
    print('Untarring...')

    # 3. make the OS-dependent .so file
    os.chdir('{}/pyRT_DISORT/disort4.0.98'.format(install_directory))
    module_name = 'disort'
    os.system('{} -m numpy.f2py -c BDREF.f DISOBRDF.f DISORT.f ERRPACK.f LAPACK.f LINPAK.f RDI1MACH.f -m {}'.format(sys.executable, module_name))

    # 4. put the .so file where python looks for its libraries
    # ... I don't know a semi-robust way to do this. Sad! ...
