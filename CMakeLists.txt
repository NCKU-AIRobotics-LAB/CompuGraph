cmake_minimum_required(VERSION 3.14)
project(COMPUGRAPH)

include(ExternalProject)
include(FetchContent)

set(COMPUGRAPH_HEADER_DIR ${COMPUGRAPH_SOURCE_DIR}/include)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The same as xtensor::optimize, but we are using manually config now
if (MSVC)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /w")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -fcompare-debug-second")
endif()

# xsimd and TBB OpenMP dependencies are automatically searched when the following is executed
set(XTENSOR_USE_XSIMD ON)
add_definitions(-DXTENSOR_USE_XSIMD)
set(XTENSOR_USE_TBB OFF)
set(XTENSOR_USE_OPENMP OFF)
find_package(TBB QUIET)
if(TBB_FOUND)
  set(XTENSOR_USE_TBB ON)
  add_definitions(-DXTENSOR_USE_TBB)
  set(tbb_libraries tbb)
endif()
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  set(XTENSOR_USE_OPENMP ON)
  add_definitions(-DXTENSOR_USE_OPENMP)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# OpenBLAS
set(BLA_VENDOR OpenBLAS)
find_package(BLAS QUIET)
find_package(LAPACK QUIET)
if(BLAS_FOUND AND LAPACK_FOUND)
  set(lapackblas_libraries ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} Threads::Threads)
  add_custom_target(openblas)
  message("Using OpenBLAS: " ${BLAS_LIBRARIES})
else()
  set(BLA_VENDOR All)
  find_package(BLAS QUIET)
  find_package(LAPACK QUIET)
  if(BLAS_FOUND AND LAPACK_FOUND)
    set(lapackblas_libraries ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} cblas Threads::Threads)
    add_custom_target(openblas)
    message("Using Default BLAS: " ${BLAS_LIBRARIES})
  else()
    message("Downloading binary OpenBLAS, if you want to build from source, please build it yourself and run CMake again.")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
      set(X86 TRUE)
    else()
      set(X86 FALSE)
    endif()
    if (X86)
      FetchContent_Declare(openblas
        URL https://github.com/xianyi/OpenBLAS/releases/download/v0.3.20/OpenBLAS-0.3.20-x86.zip
        SOURCE_DIR ${EXTERNAL_INSTALL_LOCATION}
      )
      FetchContent_MakeAvailable(openblas)
      add_custom_target(openblas)
    else()
      FetchContent_Declare(openblas
        URL https://github.com/xianyi/OpenBLAS/releases/download/v0.3.20/OpenBLAS-0.3.20-x64.zip
        SOURCE_DIR ${EXTERNAL_INSTALL_LOCATION}
      )
      FetchContent_MakeAvailable(openblas)
      add_custom_target(openblas)
    endif()
    set(lapackblas_libraries Threads::Threads)
  endif()
endif()

# XTensor
find_package(xtl QUIET)
if(xtl_FOUND)
  add_custom_target(xtl)
else()
  ExternalProject_Add(xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    DEPENDS openblas
  )
endif()
find_package(xsimd QUIET)
if(xsimd_FOUND)
  add_custom_target(xsimd)
else()
  ExternalProject_Add(xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    DEPENDS xtl
  )
endif()
find_package(xtensor QUIET)
if(xtensor_FOUND)
  add_custom_target(xtensor)
else()
  ExternalProject_Add(xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DXTENSOR_USE_XSIMD=${XTENSOR_USE_XSIMD} -DXTENSOR_USE_TBB=${XTENSOR_USE_TBB} -DXTENSOR_USE_OPENMP=${XTENSOR_USE_OPENMP}
    DEPENDS xsimd
  )
endif()
find_package(xtensor-blas QUIET)
if(xtensor-blas_FOUND)
  add_custom_target(xtensor-blas)
else()
  ExternalProject_Add(xtensor-blas
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    DEPENDS xtensor
  )
endif()
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  find_package(xtensor-python QUIET)
  if(xtensor-python_FOUND)
    add_custom_target(xtensor-python)
  else()
    ExternalProject_Add(xtensor-python
      GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-python
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DPYBIND11_PYTHON_VERSION=${PYBIND11_PYTHON_VERSION}
      DEPENDS xtensor-blas
    )
  endif()
else()
  message("Cannot find pybind11, install by pip install pybind11[global] or just it will only build the cg library.")
  add_custom_target(xtensor-python)
endif()

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(python)
