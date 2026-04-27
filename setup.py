from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "networkc._core",
            sources=["src/networkc/_core.c"],
            extra_compile_args=["-O3", "-march=native"],
        ),
        Extension(
            "networkc._dag_learn",
            sources=["src/networkc/_dag_learn.c"],
            extra_compile_args=["-O3", "-march=native"],
        ),
    ],
)
