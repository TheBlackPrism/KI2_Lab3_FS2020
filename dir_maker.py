import os
import pathlib


def make_sequential_dir(prefix):
    """Generates sequential directories with the given prefix followed by sequential numbering"""
    filepath = pathlib.Path(__file__).parent.absolute() / 'history'
    i = 0

    while os.path.exists(filepath / (prefix + ("_%04d" % i))):
        i += 1

    new_dir = filepath / (prefix + ("_%04d" % i))
    os.mkdir(new_dir)
    return new_dir
