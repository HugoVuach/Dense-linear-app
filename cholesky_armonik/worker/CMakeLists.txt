cmake_minimum_required(VERSION 3.20)
project(DagCholeskyWorker CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------
# 1. Préfixe du SDK ArmoniK (installé par le stage "sdk")
# ------------------------------------------------------------------
set(ARMONIKCPP_PREFIX "$ENV{ARMONIKCPP_PREFIX}" CACHE PATH "Prefix of ArmoniK C++ SDK install")
if(NOT ARMONIKCPP_PREFIX)
  set(ARMONIKCPP_PREFIX "/opt/armonik-cpp")
endif()

# ------------------------------------------------------------------
# 2. Protobuf / Threads
# ------------------------------------------------------------------
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

# ------------------------------------------------------------------
# 3. pkg-config pour grpc++ / Chameleon / StarPU / OpenBLAS / hwloc
# ------------------------------------------------------------------
find_program(PKG_CONFIG_EXECUTABLE pkg-config REQUIRED)

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

require_pkg(GRPCPP   grpc++)
require_pkg(CHAM     chameleon)
require_pkg(STARPU   starpu-1.4)
require_pkg(OPENBLAS openblas)
require_pkg(HWLOC    hwloc)

# ------------------------------------------------------------------
# 4. Headers du SDK ArmoniK + Protobuf
# ------------------------------------------------------------------
include_directories(${ARMONIKCPP_PREFIX}/include)
include_directories(${ARMONIKCPP_PREFIX}/include/armonik/common)  # objects.pb.h, exceptions
include_directories(${ARMONIKCPP_PREFIX}/include/armonik/worker)  # WorkerServer.h + agent_common*.pb.h
include_directories(${Protobuf_INCLUDE_DIRS})

# Cflags cumulés (gRPC/Chameleon/StarPU/BLAS/hwloc)
add_compile_options(
  ${GRPCPP_CFLAGS_LIST}
  ${CHAM_CFLAGS_LIST}
  ${STARPU_CFLAGS_LIST}
  ${OPENBLAS_CFLAGS_LIST}
  ${HWLOC_CFLAGS_LIST}
)

# ------------------------------------------------------------------
# 5. Trouver les bibliothèques du SDK ArmoniK
#     (noms possibles selon build : ArmoniK.Api.* ou armonik_*)
# ------------------------------------------------------------------
link_directories(${ARMONIKCPP_PREFIX}/lib)

function(find_armonik_lib OUT_VAR)
  set(candidates ${ARGN})
  foreach(name IN LISTS candidates)
    find_library(_lib NAMES ${name} PATHS ${ARMONIKCPP_PREFIX}/lib NO_DEFAULT_PATH)
    if(_lib)
      set(${OUT_VAR} "${_lib}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${OUT_VAR} "NOTFOUND" PARENT_SCOPE)
endfunction()

# Worker (TaskHandler, ArmoniKWorker, WorkerServer…)
find_armonik_lib(ARMONIK_WORKER
  ArmoniK.Api.Worker armonik_worker armonikapiworker ArmoniKWorker
)
# Common (utils::Configuration, exceptions, objects.pb.o…)
find_armonik_lib(ARMONIK_COMMON
  ArmoniK.Api.Common armonik_common armonikapicommon ArmoniKCommon
)
# Logger (Logger, writers/formatters)
find_armonik_lib(ARMONIK_LOGGER
  ArmoniK.Api.Logger armonik_logger armonikapilogger ArmoniKLogger
)

if(NOT ARMONIK_WORKER OR NOT EXISTS "${ARMONIK_WORKER}")
  message(FATAL_ERROR
    "Bibliothèque Worker du SDK introuvable dans ${ARMONIKCPP_PREFIX}/lib.\n"
    "Contenu dispo :\n  ${ARMONIKCPP_PREFIX}/lib")
endif()
if(NOT ARMONIK_COMMON OR NOT EXISTS "${ARMONIK_COMMON}")
  message(FATAL_ERROR "Bibliothèque Common du SDK introuvable dans ${ARMONIKCPP_PREFIX}/lib.")
endif()
if(NOT ARMONIK_LOGGER OR NOT EXISTS "${ARMONIK_LOGGER}")
  message(FATAL_ERROR "Bibliothèque Logger du SDK introuvable dans ${ARMONIKCPP_PREFIX}/lib.")
endif()

message(STATUS "ArmoniK worker lib  : ${ARMONIK_WORKER}")
message(STATUS "ArmoniK common lib  : ${ARMONIK_COMMON}")
message(STATUS "ArmoniK logger lib  : ${ARMONIK_LOGGER}")


# ------------------------------------------------------------------
# 6. Exécutable
# ------------------------------------------------------------------
add_executable(DagCholeskyWorker
  src/worker_distib.cpp
)


# Link de base
target_link_libraries(DagCholeskyWorker
  ${Protobuf_LIBRARIES}
  Threads::Threads
)

# Ajouter les libs ArmoniK (doivent arriver avant/avec gRPC/protobuf)
target_link_libraries(DagCholeskyWorker PRIVATE
  ${ARMONIK_WORKER}
  ${ARMONIK_COMMON}
  ${ARMONIK_LOGGER}
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


# ------------------------------------------------------------------
# 7. RPATH pour retrouver les .so au runtime
# ------------------------------------------------------------------
set_target_properties(DagCholeskyWorker PROPERTIES
  BUILD_RPATH   "${ARMONIKCPP_PREFIX}/lib"
  INSTALL_RPATH "${ARMONIKCPP_PREFIX}/lib"
)

