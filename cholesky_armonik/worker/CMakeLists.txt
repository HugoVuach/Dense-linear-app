cmake_minimum_required(VERSION 3.20)
project(DagCholeskyWorker CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Option : emplacement d'install du SDK C++ ArmoniK (donné par le stage sdk)
set(ARMONIKCPP_PREFIX "$ENV{ARMONIKCPP_PREFIX}" CACHE PATH "Prefix of ArmoniK C++ SDK install")
if(NOT ARMONIKCPP_PREFIX)
  set(ARMONIKCPP_PREFIX "/opt/armonik-cpp")
endif()

# Protobuf / gRPC
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

# pkg-config requis pour récupérer grpc++ et libs Chameleon/StarPU/OpenBLAS/hwloc
find_program(PKG_CONFIG_EXECUTABLE pkg-config REQUIRED)

# Helper pour exiger une lib via pkg-config
macro(require_pkg name module)
  execute_process(COMMAND ${PKG_CONFIG_EXECUTABLE} --exists ${module} RESULT_VARIABLE HAVE_${name})
  if(NOT HAVE_${name} EQUAL 0)
    message(FATAL_ERROR "${module} introuvable via pkg-config")
  endif()
  execute_process(COMMAND ${PKG_CONFIG_EXECUTABLE} --cflags ${module} OUTPUT_VARIABLE ${name}_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${PKG_CONFIG_EXECUTABLE} --libs   ${module} OUTPUT_VARIABLE ${name}_LIBS   OUTPUT_STRIP_TRAILING_WHITESPACE)
  separate_arguments(${name}_CFLAGS_LIST NATIVE_COMMAND "${${name}_CFLAGS}")
  separate_arguments(${name}_LIBS_LIST   NATIVE_COMMAND "${${name}_LIBS}")
endmacro()

# Libs nécessaires
require_pkg(GRPCPP   grpc++)
require_pkg(CHAM     chameleon)
require_pkg(STARPU   starpu-1.4)
require_pkg(OPENBLAS openblas)
require_pkg(HWLOC    hwloc)

# Include du SDK C++ ArmoniK (headers Worker/*, utils/*, exceptions/*, objects.pb.h…)
include_directories(${ARMONIKCPP_PREFIX}/include)
include_directories(${Protobuf_INCLUDE_DIRS})

# Cflags cumulés (gRPC/Chameleon/StarPU/BLAS/hwloc)
add_compile_options(
  ${GRPCPP_CFLAGS_LIST}
  ${CHAM_CFLAGS_LIST}
  ${STARPU_CFLAGS_LIST}
  ${OPENBLAS_CFLAGS_LIST}
  ${HWLOC_CFLAGS_LIST}
)

# Ton code worker (fichier que tu m'as donné)
add_executable(DagCholeskyWorker
  src/worker.cpp
)

# Link de base
target_link_libraries(DagCholeskyWorker
  ${Protobuf_LIBRARIES}
  Threads::Threads
)

# Ajout des libs via flags pkg-config (ordre conservé)
target_link_options(DagCholeskyWorker PRIVATE
  ${GRPCPP_LIBS_LIST}
  ${CHAM_LIBS_LIST}
  ${STARPU_LIBS_LIST}
  ${OPENBLAS_LIBS_LIST}
  ${HWLOC_LIBS_LIST}
  -ldl -lrt
)

# (Option) emplacement des libs ArmoniK si des .so/.a existent dans le prefix
link_directories(${ARMONIKCPP_PREFIX}/lib)
