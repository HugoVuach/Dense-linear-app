cmake_minimum_required(VERSION 3.22)
project(armonik_cholesky_client VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
find_package(ArmoniK.Api.Client REQUIRED) # fournit aussi ArmoniK.Api.Common

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_executable(${PROJECT_NAME}
  ${SRC_DIR}/client_distrib.cpp
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC ArmoniK.Api.Common ArmoniK.Api.Client Threads::Threads
)

target_include_directories(${PROJECT_NAME}
  PUBLIC ${SRC_DIR} "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
