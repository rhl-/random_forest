## The project
project(rf CXX)
## The minimum version required (wasn't tested with on older version)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
add_definitions(-std=c++11)
add_executable(tree_test test_tree.cpp)
