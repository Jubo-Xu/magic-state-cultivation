"""
Build the _clustering_cpp pybind11 extension.

Run from this directory (prior_gap_estimation_cpp/):
    pip install pybind11          # once
    python setup.py build_ext --inplace

This produces:
    prior_gap_estimation_cpp/_clustering_cpp.cpython-<ver>-<arch>.so
"""

import os
from setuptools import Extension, setup

import pybind11

_src = os.path.join(os.path.dirname(__file__), "src")

ext = Extension(
    "_clustering_cpp",
    sources=[
        os.path.join(_src, "bindings.cpp"),
        os.path.join(_src, "clustering.cpp"),
        os.path.join(_src, "clustering_overgrow_batch.cpp"),
        os.path.join(_src, "prior_gap_estimator.cpp"),
        os.path.join(_src, "incremental_rref.cpp"),
        os.path.join(_src, "gf2matrix.cpp"),
    ],
    include_dirs=[pybind11.get_include(), _src],
    extra_compile_args=[
        "-std=c++17",
        "-O3",
        "-mavx2",
        "-march=native",
        "-fvisibility=hidden",   # recommended for pybind11 extensions
    ],
    language="c++",
)

setup(
    name="clustering_cpp",
    version="0.1.0",
    ext_modules=[ext],
)
