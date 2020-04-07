cmake_minimum_required(VERSION 2.8.9)

add_library(
  Packets
  netikat.h
  netikat.cpp
  linear_algebra_utilities.h
  linear_algebra_utilities.ipp
  linear_algebra_utilities.cpp
  utils.h
  utils.cpp
)
target_link_libraries(Packets eigen)

add_executable(
  benchmark
  benchmark.cpp
)
target_include_directories(benchmark PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(benchmark Packets)

add_executable(
  run
  main.cpp
)
target_include_directories(run PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(run Packets)

option(PACKAGE_TESTS "Build the tests" ON)

if(PACKAGE_TESTS)
  enable_testing()
  add_subdirectory(
    tests
  )
endif()
