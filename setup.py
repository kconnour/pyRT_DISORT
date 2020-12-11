import glob
import os
import setuptools
import sys


class SetupDISORT:
    def __init__(self, install_disort=True):
        self.install_disort = install_disort
        self.project_path = os.path.dirname(os.path.realpath(__file__))

        if self.install_disort:
            self.disort_folder_name = 'disort4.0.99'
            self.module_name = 'disort'
            self.compile_disort_so_file()
            self.so_file_name = self.get_so_file_name()
            self.move_so_file_up_one_directory()
        self.setup_package()

    def compile_disort_so_file(self):
        os.chdir(os.path.join(self.project_path, self.disort_folder_name))
        os.system(
            f'{sys.executable} -m numpy.f2py -c BDREF.f DISOBRDF.f DISORT.f ERRPACK.f LAPACK.f LINPAK.f RDI1MACH.f '
            f'-m {self.module_name}')

    @staticmethod
    def get_so_file_name():
        disort_binary_filename = glob.glob('*.so')[0]
        return disort_binary_filename

    def move_so_file_up_one_directory(self):
        os.rename(os.path.join(os.path.join(self.project_path, self.disort_folder_name), self.so_file_name),
                  os.path.join(self.project_path, self.so_file_name))

    def setup_package(self):
        os.chdir(self.project_path)
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
                'astropy>=4.1',
                'numba>=0.51.2',
                'numpy>=1.19.1',
                'pandas>=1.1.1',
                'pdoc3>=0.9.1'
            ],
            # I have no clue why I need to go up a directory when I'm already in the correct directory
            # Also, this installs any .so files present, so it installs disort if it's present, but otherwise nothing
            package_data={'': ['../*.so']}
        )


if __name__ == '__main__':
    SetupDISORT(install_disort=False)
