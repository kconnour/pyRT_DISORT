from pathlib import Path
import setuptools
import sys

from numpy import f2py


# Define project variables
project_path = Path(__file__).resolve().parent
disort_directory = project_path.joinpath('disort4.0.99')
module_name = 'disort'
fortran_source_filenames = ['BDREF.f', 'DISOBRDF.f', 'ERRPACK.f', 'LAPACK.f', 'LINPAK.f', 'RDI1MACH.f']

# Currently, Python calls setup.py 2 times! I don't really know why but the first time is to generate egg info.
#  So, only execute the disort compilation code once to avoid unnecessary work and generating redundant .so files
if sys.argv[1] == 'egg_info':
    # Compile disort into one file
    fortran_paths = [disort_directory.joinpath(f) for f in fortran_source_filenames]
    with open(disort_directory.joinpath('DISORT.f')) as disort_module:
        f2py.compile(disort_module.read(), modulename=module_name, extra_args=fortran_paths)

    # Rename disort
    binary_file = list(project_path.glob('*.so'))[0]
    binary_file.rename(project_path / 'disort.so')

# Install the project
setuptools.setup()

# Cleanup. bdist_wheel is executed last.
if sys.argv[1] == 'bdist_wheel':
    def delete_folder(path: Path):
        for sub in path.iterdir():
            if sub.is_dir():
                delete_folder(sub)
            else:
                sub.unlink()
        path.rmdir()

    Path(project_path / 'disort.so').unlink()
    delete_folder(Path(project_path / 'build'))
    delete_folder(Path(project_path / 'pyRT_DISORT.egg-info'))
