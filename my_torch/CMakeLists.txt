cmake_minimum_required(VERSION 3.20)

option(USE_CUDA "Enable CUDA backend" OFF)

if(USE_CUDA)
    project(my_torch CXX CUDA)
else()
    project(my_torch CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# ---- sources always compiled ----------------------------------------
set(SOURCES
    main.cpp
    src/autograd.cpp
    src/nn.cpp
)

# ---- backend selection ----------------------------------------------
if(USE_CUDA)
    list(APPEND SOURCES src/kernels_cuda.cu)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(my_torch PRIVATE USE_CUDA)
else()
    list(APPEND SOURCES src/kernels_cpu.cpp)
endif()

add_executable(my_torch ${SOURCES})

if(USE_CUDA)
    target_compile_definitions(my_torch PRIVATE USE_CUDA)
    set_target_properties(my_torch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

find_package(Threads REQUIRED)
target_link_libraries(my_torch PRIVATE Threads::Threads)

target_compile_options(my_torch PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/W4>>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O3>>
)
