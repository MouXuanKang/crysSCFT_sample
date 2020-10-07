cmake_minimum_required(VERSION 3.10)

project(crysSCFT_sample>)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

add_compile_options(-O3 -axCORE-AVX2 -restrict -ansi-alias -fast -ipo -qopenmp)

add_executable(crysSCFT_sample crysSCFT_sample.cpp)
target_link_libraries(crysSCFT_sample PUBLIC "-lfftw3 -lfftw3_threads -qopenmp -liomp5 -lstdc++ -lm -lpthread -ldl")
