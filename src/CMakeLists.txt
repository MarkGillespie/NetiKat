cmake_minimum_required(VERSION 2.8.9)

# add_library(
#   my_assert
#   my_assert.cpp
#   my_assert.h
# )

add_executable(
  run
  main.cpp
)
target_include_directories(run PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
# target_link_libraries(run polyscope geometry-central)

option(PACKAGE_TESTS "Build the tests" ON)

if(PACKAGE_TESTS)
  enable_testing()
  add_subdirectory(
    tests
  )
endif()
