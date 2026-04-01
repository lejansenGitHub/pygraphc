from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "connected_component._core",
            sources=["src/connected_component/_core.c"],
            extra_compile_args=["-O3", "-march=native"],
        ),
    ],
)
