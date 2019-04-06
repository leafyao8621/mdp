from distutils.core import setup, Extension
setup(name='mdp', version='1.0',  \
      ext_modules=[Extension('mdp', ["matrix.c",
                                          "mdp.c",
                                          "mod.c",
                                          "util.c"],
                             extra_compile_args = ["-fopenmp"],
                             extra_link_args = ["-fopenmp"])])
