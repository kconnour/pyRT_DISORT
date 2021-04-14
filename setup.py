import os
import setuptools
from numpy import f2py


# TODO: see if there's a way to check if the .so file is up to date. If so,
#  don't bother reinstalling it. Otherwise, do install it.
class SetupDISORT:
    def __init__(self) -> None:
        self.__project_path = self.__get_project_path()
        self.__install_disort()
        setuptools.setup()

    @staticmethod
    def __get_project_path() -> str:
        return os.path.dirname(os.path.realpath(__file__))

    def __install_disort(self) -> None:
        # Alternative:
        # https://stackoverflow.com/questions/64950460/link-f2py-generated-so-file-in-a-python-package-using-setuptools
        # https://numpy.org/devdocs/f2py/distutils.html
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


SetupDISORT()
