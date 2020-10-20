import os

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
module_path = os.path.join(package_path, 'pyRT_DISORT')
os.system('pdoc --html {}'.format(module_path))
