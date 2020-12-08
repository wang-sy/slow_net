cmake_minimum_required(VERSION 3.17)
project(slow_net)
find_package(Eigen3)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
set(CMAKE_CXX_STANDARD 14)

add_executable(slow_net main.cpp src/base/neuron_based.cc src/base/neuron_based.h src/includes.h src/default_neuron.cc src/default_neuron.h test/default_neuron_test.cc src/base/active_functions.cc src/base/active_functions.h)