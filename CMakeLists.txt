cmake_minimum_required(VERSION 3.5)

project(opencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/kernels/*.cl")

add_executable(opencv main.cpp Setup.cpp Setup.h)
FILE(COPY kernels images DESTINATION "${CMAKE_BINARY_DIR}")

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv OpenCL::OpenCL ${OpenCV_LIBS} -fopenmp)
