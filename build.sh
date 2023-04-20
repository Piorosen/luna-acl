#!/bin/bash

scons Werror=0 arch=arm64-v8a neon=1 opencl=0 embed_kernels=1 extra_cxx_flags="-fPIC" -j$(nproc)
