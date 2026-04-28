from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "pygraphc._core",
            sources=["src/pygraphc/_core.c"],
            extra_compile_args=["-O3", "-march=native"],
        ),
        Extension(
            "pygraphc._dag_learn",
            sources=["src/pygraphc/_dag_learn.c"],
            extra_compile_args=["-O3", "-march=native"],
        ),
    ],
)
