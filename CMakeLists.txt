## The project
project(RF CXX)
## The minimum version required (wasn't tested with on older version)
cmake_minimum_required(VERSION 2.8)

include_directories($ENV{HOME}/Dev/ayasdi/math/include)
include_directories($ENV{HOME}/code/math/include)
include_directories($ENV{HOME}/code/math/lapack/include)
include_directories($ENV{HOME}/Dev/ayasdi/math/lapack/include)
include_directories($ENV{HOME}/Dev/random_forest)
include_directories(${CMAKE_CURRENT_LIST_DIR})

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
add_definitions(-std=c++11)
add_library(CatchObj OBJECT tests/catch.cpp)
add_executable(rf_test tests/test_tree.cpp $<TARGET_OBJECTS:CatchObj>)
add_executable(rf rf.cpp)
