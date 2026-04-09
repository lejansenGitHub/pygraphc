from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "cgraph._core",
            sources=["src/cgraph/_core.c"],
            extra_compile_args=["-O3", "-march=native"],
        ),
        Extension(
            "cgraph._dag_learn",
            sources=["src/cgraph/_dag_learn.c"],
            extra_compile_args=["-O3", "-march=native"],
        ),
    ],
)
