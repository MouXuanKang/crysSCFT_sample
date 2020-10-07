#!/bin/bash
dst=Build
cmake3 -Hsrc -B${dst} -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc 
cmake3 --build ./${dst}
./${dst}/crysSCFT_sample