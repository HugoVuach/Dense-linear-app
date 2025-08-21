cmake_minimum_required(VERSION 3.22)
project(armonik_cpp_cholesky VERSION 0.1 LANGUAGES C CXX)

find_package(Threads)
find_package(ArmoniK.Api.Worker REQUIRED)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE SRC_FILES ${SRC_DIR}/*.cpp)
file(GLOB_RECURSE HEADER_FILES ${SRC_DIR}/*.h)

add_executable(${PROJECT_NAME} ${SRC_FILES} ${HEADER_FILES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_link_libraries(${PROJECT_NAME} PUBLIC ArmoniK.Api.Common ArmoniK.Api.Worker)

# Ajoute les includes du worker + CUDA headers
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${SRC_DIR}
    /usr/local/cuda/include
)


