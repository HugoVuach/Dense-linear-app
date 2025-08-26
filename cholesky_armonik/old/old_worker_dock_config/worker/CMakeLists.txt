cmake_minimum_required(VERSION 3.20)
project(DagCholeskyWorker LANGUAGES CXX)

# ------------------------------------------------------------------
# 0. Options C++
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ------------------------------------------------------------------
# 1. Préfixe du SDK ArmoniK (fourni par le sample)
# ------------------------------------------------------------------
set(ARMONIKCPP_PREFIX "$ENV{ARMONIKCPP_PREFIX}" CACHE PATH "Prefix of ArmoniK C++ SDK install")
if(NOT ARMONIKCPP_PREFIX OR ARMONIKCPP_PREFIX STREQUAL "")
  set(ARMONIKCPP_PREFIX "/opt/armonik-cpp")
endif()

# ------------------------------------------------------------------
# 2. Verrouillage de pkg-config sur le préfixe ArmoniK
#    -> Empêche toute résolution vers /usr, /usr/local, etc.
# ------------------------------------------------------------------
find_package(PkgConfig REQUIRED)

set(_pcdirs
  "${ARMONIKCPP_PREFIX}/lib/pkgconfig"
  "${ARMONIKCPP_PREFIX}/lib64/pkgconfig"
  "${ARMONIKCPP_PREFIX}/share/pkgconfig")

# Rejoint _pcdirs avec ':' sans utiliser list(JOIN ...)
set(_pcdir_join "")
foreach(_d ${_pcdirs})
  if(EXISTS "${_d}")
    if(_pcdir_join)
      set(_pcdir_join "${_pcdir_join}:${_d}")
    else()
      set(_pcdir_join "${_d}")
    endif()
  endif()
endforeach()

# Purge les chemins par défaut et remplace par le préfixe SDK
set(ENV{PKG_CONFIG_DIR}     "")
set(ENV{PKG_CONFIG_PATH}    "")
set(ENV{PKG_CONFIG_LIBDIR}  "${_pcdir_join}")

# ------------------------------------------------------------------
# 3. Dépendances via pkg-config (STRICT: uniquement dans le préfixe)
# ------------------------------------------------------------------
# gRPC++ (secure/unsecure selon build du SDK)
pkg_search_module(GRPCPP REQUIRED IMPORTED_TARGET grpc++ grpc++_unsecure)

# Protobuf (certaines distros nomment 'libprotobuf')
pkg_search_module(PROTO   REQUIRED IMPORTED_TARGET protobuf libprotobuf)

# Chameleon
pkg_search_module(CHAM    REQUIRED IMPORTED_TARGET chameleon)

# StarPU (1.4/1.3, avec/sans OpenMP)
pkg_search_module(STARPU  REQUIRED IMPORTED_TARGET
  starpu-1.4-openmp starpu-1.4
  starpu-1.3-openmp starpu-1.3)

# OpenBLAS (32/64)
pkg_search_module(OPENBLAS REQUIRED IMPORTED_TARGET openblas openblas64_)

# hwloc
pkg_search_module(HWLOC   REQUIRED IMPORTED_TARGET hwloc)

# Threads (ne provient pas du SDK, mais interface CMake portable)
find_package(Threads REQUIRED)

# ------------------------------------------------------------------
# 4. Headers du SDK ArmoniK
# ------------------------------------------------------------------
set(ARMONIK_INCLUDE_DIRS
  "${ARMONIKCPP_PREFIX}/include"
  "${ARMONIKCPP_PREFIX}/include/armonik/common"
  "${ARMONIKCPP_PREFIX}/include/armonik/worker"
)

# ------------------------------------------------------------------
# 5. Source
# ------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/worker_distrib.cpp")
  set(WORKER_SRC src/worker_distrib.cpp)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/worker_distib.cpp")
  set(WORKER_SRC src/worker_distib.cpp)
else()
  message(FATAL_ERROR "Aucun fichier source trouvé : src/worker_distrib.cpp (ou src/worker_distib.cpp).")
endif()

add_executable(DagCholeskyWorker ${WORKER_SRC})

target_include_directories(DagCholeskyWorker PRIVATE
  ${ARMONIK_INCLUDE_DIRS}
)

# ------------------------------------------------------------------
# 6. Libs ArmoniK — Worker + Common uniquement dans le préfixe SDK
# ------------------------------------------------------------------
function(find_armonik_lib OUT_VAR)
  # Passe une liste de noms candidats après OUT_VAR
  set(_candidates ${ARGN})
  foreach(_n IN LISTS _candidates)
    find_library(_lib
      NAMES "${_n}"
      PATHS "${ARMONIKCPP_PREFIX}/lib" "${ARMONIKCPP_PREFIX}/lib64"
      NO_DEFAULT_PATH
    )
    if(_lib)
      set(${OUT_VAR} "${_lib}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${OUT_VAR} "NOTFOUND" PARENT_SCOPE)
endfunction()

# Worker
find_armonik_lib(ARMONIK_WORKER_LIB
  ArmoniK.Api.Worker      # <-- nom réel du sample
  ArmoniKWorker
  armonik_worker
  armonikapiworker
)
if(ARMONIK_WORKER_LIB STREQUAL "NOTFOUND")
  file(GLOB _libs "${ARMONIKCPP_PREFIX}/lib/*" "${ARMONIKCPP_PREFIX}/lib64/*")
  message(FATAL_ERROR "Librairie ArmoniK.Api.Worker introuvable dans ${ARMONIKCPP_PREFIX}/lib(64).\nContenu détecté:\n${_libs}")
endif()
message(STATUS "ArmoniK worker lib  : ${ARMONIK_WORKER_LIB}")

# Common (logger, configuration…)
find_armonik_lib(ARMONIK_COMMON_LIB
  ArmoniK.Api.Common      # <-- nom réel du sample
  ArmoniKCommon
  armonik_common
  armonikapicommon
)
if(ARMONIK_COMMON_LIB STREQUAL "NOTFOUND")
  file(GLOB _libs "${ARMONIKCPP_PREFIX}/lib/*" "${ARMONIKCPP_PREFIX}/lib64/*")
  message(FATAL_ERROR "Librairie ArmoniK.Api.Common introuvable dans ${ARMONIKCPP_PREFIX}/lib(64).\nContenu détecté:\n${_libs}")
endif()
message(STATUS "ArmoniK common lib  : ${ARMONIK_COMMON_LIB}")

# ------------------------------------------------------------------
# 7. Lien — ordre important pour les libs statiques
# ------------------------------------------------------------------
set(EXTRA_SYS_LIBS)

if(UNIX AND NOT APPLE)
  find_library(DL_LIB dl)
  if(DL_LIB)
    list(APPEND EXTRA_SYS_LIBS "${DL_LIB}")
  endif()

  find_library(RT_LIB rt)
  if(RT_LIB)
    list(APPEND EXTRA_SYS_LIBS "${RT_LIB}")
  endif()
endif()

# Pour les toolchains GNU/Clang, on peut entourer toutes les libs d’un group
# pour résoudre d’éventuelles dépendances circulaires.
if(UNIX AND NOT APPLE)
  target_link_options(DagCholeskyWorker PRIVATE -Wl,--start-group)
endif()

target_link_libraries(DagCholeskyWorker PRIVATE
  # D'abord nos libs ArmoniK (Worker dépend de Common)
  ${ARMONIK_WORKER_LIB}
  ${ARMONIK_COMMON_LIB}

  # Puis les dépendances du SDK
  PkgConfig::GRPCPP
  PkgConfig::PROTO
  PkgConfig::CHAM
  PkgConfig::STARPU
  PkgConfig::OPENBLAS
  PkgConfig::HWLOC
  Threads::Threads
  ${EXTRA_SYS_LIBS}
)

if(UNIX AND NOT APPLE)
  target_link_options(DagCholeskyWorker PRIVATE -Wl,--end-group)
endif()

# GCC < 9 : std::filesystem dédié
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(DagCholeskyWorker PRIVATE stdc++fs)
endif()

# ------------------------------------------------------------------
# 8. RPATH limité au préfixe du sample (exécution et installation)
# ------------------------------------------------------------------
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set_target_properties(DagCholeskyWorker PROPERTIES
  BUILD_RPATH   "$ORIGIN;${ARMONIKCPP_PREFIX}/lib;${ARMONIKCPP_PREFIX}/lib64"
  INSTALL_RPATH "$ORIGIN;${ARMONIKCPP_PREFIX}/lib;${ARMONIKCPP_PREFIX}/lib64"
)