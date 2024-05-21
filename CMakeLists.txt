cmake_minimum_required(VERSION 3.5)

project(opencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXE_LINKER_FLAGS "-labsl_log_internal_check_op -labsl_log_internal_message")

add_executable(opencv main.cpp )
FILE(COPY image_kernel.cl images DESTINATION "${CMAKE_BINARY_DIR}")

find_package(OpenCL REQUIRED)
find_package(OpenCLHeaders REQUIRED)
find_package(OpenCLHeadersCpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv OpenCL::Headers OpenCL::OpenCL OpenCL::HeadersCpp ${OpenCV_LIBS} -fopenmp)
