import os
import tarfile


with tarfile.open('pyRT_DISORT.tar.gz', 'w:gz') as tar_handle:
    tar_handle.add(os.getcwd(), arcname='pyRT_DISORT')

tar_handle.close()
