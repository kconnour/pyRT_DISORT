# Built-in imports
import os

# 3rd-party imports
import pdoc

# Local imports
from path import get_project_path

# TODO: this fails if I run it from command line
# TODO: can I remove list of one item (self.module)? (line 19)
# TODO: This is not OS independent---only works on Mac/Linux
# TODO: Remove tests from showing up in the doc
# TODO: This shows sub-packages as sub-modules... can I fix that?
# TODO: This code is slightly messy


class Documentation:
    """ A Documentation object can create documentation for pyRT_DISORT. """
    def __init__(self) -> None:
        self.module = [pdoc.Module('pyRT_DISORT', context=pdoc.Context())]
        pdoc.link_inheritance(pdoc.Context())
        self.html_path = self.__get_path_where_to_put_html_doc()

    @staticmethod
    def __get_path_where_to_put_html_doc():
        return os.path.abspath(os.path.join(get_project_path(), 'doc/html'))

    def make_doc(self) -> None:
        """Make documentation

        Returns
        -------
        None

        """
        for mod in self.module:
            for module_name, html, is_folder in self.__make_recursive_html(mod):
                if 'test' in module_name:
                    continue
                self.__make_folder(module_name) if is_folder else False
                abs_path = self.__make_html_name(module_name, is_folder)
                self.__write_html_to_computer(abs_path, html)

    def __make_recursive_html(self, module: pdoc.Module):
        yield module.name, module.html(latex_math=True), bool(module.submodules())
        for submodules in module.submodules():
            yield from self.__make_recursive_html(submodules)

    def __make_folder(self, module_name: str) -> None:
        folder_name = module_name.replace('.', '/')
        folder_path = os.path.join(self.html_path, folder_name)
        self.make_folder_if_nonexistent(folder_path)

    @staticmethod
    def make_folder_if_nonexistent(folder_path: str) -> None:
        try:
            os.mkdir(folder_path)
        except FileExistsError:
            pass

    def __make_html_name(self, module_name: str, is_folder: bool) -> str:
        name = module_name.replace('.', '/')
        relative_name = name + '/index' if is_folder else name
        return os.path.join(self.html_path, relative_name)

    @staticmethod
    def __write_html_to_computer(abs_path: str, html: str) -> None:
        with open(f'{abs_path}.html', 'w+') as html_file:
            html_file.write(html)


if __name__ == '__main__':
    Documentation().make_doc()
