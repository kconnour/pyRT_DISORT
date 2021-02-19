import glob
import os
import setuptools
import sys


class SetupDISORT:
    def __init__(self, install_disort: bool = False) -> None:
        self.__project_path = self.__get_project_path()
        if install_disort:
            self.__install_disort()
        self.__setup_package()

    @staticmethod
    def __get_project_path() -> str:
        return os.path.dirname(os.path.realpath(__file__))

    def __install_disort(self) -> None:
        folder_name = 'disort4.0.99'
        module_name = 'disort'
        self.__compile_disort_binary_file(folder_name, module_name)
        binary_filename = self.__get_binary_filename()
        self.__move_binary_file_up_one_directory(binary_filename, folder_name)

    def __compile_disort_binary_file(self, folder_name: str, module_name: str) \
            -> None:
        os.chdir(os.path.join(self.__project_path, folder_name))
        os.system(
            f'{sys.executable} -m numpy.f2py -c BDREF.f DISOBRDF.f DISORT.f '
            f'ERRPACK.f LAPACK.f LINPAK.f RDI1MACH.f -m {module_name}')

    @staticmethod
    def __get_binary_filename() -> str:
        return glob.glob('*.so')[0]

    def __move_binary_file_up_one_directory(self, binary_filename: str,
                                            folder_name: str) -> None:
        os.rename(os.path.join(os.path.join(self.__project_path, folder_name),
                               binary_filename),
                  os.path.join(self.__project_path, binary_filename))

    def __setup_package(self) -> None:
        os.chdir(self.__project_path)
        setuptools.setup(
            name='pyRT_DISORT',
            version='0.0.1',
            description='Make radiative transfer more accessible to the '
                        'yearning masses',
            url='https://github.com/kconnour/pyRT_DISORT',
            author='kconnour',
            packages=setuptools.find_packages(),
            include_package_data=True,
            python_requires='>=3.9',
            install_requires=[
                'astropy>=4.1',
                'numpy>=1.20.0',
                'pandas>=1.2.1',
                'pdoc3>=0.9.1',
                'scipy>=1.6.0'
            ],
            package_data={'': ['../*.so']}
        )


if __name__ == '__main__':
    SetupDISORT(install_disort=True)
