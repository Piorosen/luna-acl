#!/bin/bash

scons Werror=0 arch=arm64-v8a neon=1 opencl=0 examples=0 embed_kernels=1 exceptions=0 extra_cxx_flags="-fPIC" -j$(nproc)

sshpass -p "linaro" scp ./build/*.so linaro@192.168.0.229:/home/linaro/Desktop/chacha