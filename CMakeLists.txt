cmake_minimum_required(VERSION 3.14)
project(ClothSimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("/home/bingbing/Physics_Simulation/shell/polyscope")

find_package(Eigen3 REQUIRED)

add_executable(${PROJECT_NAME} test.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE polyscope Eigen3::Eigen)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)
