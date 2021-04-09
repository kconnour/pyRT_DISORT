import os
import setuptools


class SetupDISORT:
    def __init__(self, install_disort: bool = True) -> None:
        self.__project_path = self.__get_project_path()
        if install_disort:
            self.__install_disort()
        self.__setup_package()

    @staticmethod
    def __get_project_path() -> str:
        return os.path.dirname(os.path.realpath(__file__))

    def __install_disort(self) -> None:
        from numpy import f2py
        folder_name = 'disort4.0.99'
        module_name = 'disort'

        disort_source_dir = os.path.join(self.__project_path, folder_name)
        mods = ['BDREF.f', 'DISOBRDF.f', 'ERRPACK.f', 'LAPACK.f',
                'LINPAK.f', 'RDI1MACH.f']
        paths = [os.path.join(disort_source_dir, m) for m in mods]
        # I'm disgusted to say I'm adding a comment. I want to compile DISORT.f
        #  as a module, and I can do that by adding the other modules it needs
        #  in extra_args (this wasn't clear in f2py documentation).
        with open(os.path.join(disort_source_dir, 'DISORT.f')) as mod:
            f2py.compile(mod.read(), modulename=module_name, extra_args=paths)

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
                'astropy>=4.2',
                'numpy>=1.20.1',
                'pandas>=1.2.3',
                'scipy>=1.6.1',
                'wheel>=0.36.2'
            ],
            # If you want to test pyRT_DISORT, these are needed:
            # 'pytest>=6.2.2'

            # If you want to make documentation, these are needed:
            # 'Sphinx>=3.4.3',
            # 'sphinx_autodoc_typehints',
            # 'sphinx-rtd-theme'
            package_data={'': ['../*.so']}
        )


if __name__ == '__main__':
    SetupDISORT(install_disort=True)
