cmake_minimum_required(VERSION 3.14)
project(COMPUGRAPH)

include(ExternalProject)

set(COMPUGRAPH_HEADER_DIR ${COMPUGRAPH_SOURCE_DIR}/include)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The same as xtensor::optimize, but we are using manually config now
if (MVSC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math")
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
  message(Using OpenBLAS: ${BLAS_LIBRARIES})
else()
  set(BLA_VENDOR All)
  find_package(BLAS QUIET)
  find_package(LAPACK QUIET)
  if(BLAS_FOUND AND LAPACK_FOUND)
    set(lapackblas_libraries ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} cblas Threads::Threads)
    add_custom_target(openblas)
    message(Using Default BLAS: ${BLAS_LIBRARIES})
  else()
    if (DEFINED WIN32)
      ExternalProject_Add(openblas
        URL http://github.com/xianyi/OpenBLAS/archive/v0.3.6.tar.gz
        CMAKE_GENERATOR "MSYS Makefiles"
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE 1
        BUILD_COMMAND make -j${JJ} CMAKE_GNUtoMS=ON BUILD_SHARED_LIBS=ON CMAKE_BUILD_TYPE=Release
      )
    else()
      ExternalProject_Add(openblas
        URL http://github.com/xianyi/OpenBLAS/archive/v0.3.6.tar.gz
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE 1
        BUILD_COMMAND make -j${JJ} BUILD_SHARED_LIBS=ON CMAKE_BUILD_TYPE=Release
        INSTALL_COMMAND make PREFIX=${EXTERNAL_INSTALL_LOCATION} install
      )
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

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(python)
