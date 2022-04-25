cmake_minimum_required(VERSION 3.14)
project(COMPUGRAPH)

include(ExternalProject)

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
endif()
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  set(XTENSOR_USE_OPENMP ON)
  add_definitions(-DXTENSOR_USE_OPENMP)
endif()

set(COMPUGRAPH_HEADER_DIR ${COMPUGRAPH_SOURCE_DIR}/include)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} 
)
ExternalProject_Add(xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
		DEPENDS xtl
)
ExternalProject_Add(xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DXTENSOR_USE_XSIMD=${XTENSOR_USE_XSIMD} -DXTENSOR_USE_TBB=${XTENSOR_USE_TBB} -DXTENSOR_USE_OPENMP=${XTENSOR_USE_OPENMP}
		DEPENDS xsimd
)
ExternalProject_Add(xtensor-blas
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
		DEPENDS xtensor
)
ExternalProject_Add(xtensor-python
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-python
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DPYBIND11_PYTHON_VERSION=${PYBIND11_PYTHON_VERSION}
		DEPENDS xtensor-blas
)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
	set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

add_subdirectory(src)
add_subdirectory(test)

find_package(pybind11 CONFIG REQUIRED)
include_directories(${COMPUGRAPH_HEADER_DIR})
FILE(GLOB PYBIND_SOURCES python/**.cpp)
find_package(Python3 ${PYBIND11_PYTHON_VERSION} COMPONENTS NumPy REQUIRED)
pybind11_add_module(compugraph ${PYBIND_SOURCES})
target_link_libraries(compugraph PUBLIC cg Python3::NumPy)

# for xsimd
if(WIN32)
	set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
	if(HAVE_AVX2_EXTENSIONS AND NOT MSVC_VERSION LESS 1800)
		target_compile_options(compugraph PRIVATE /arch:AVX2)
	elseif(HAVE_AVX_EXTENSIONS AND NOT MSVC_VERSION LESS 1600)
		target_compile_options(compugraph PRIVATE /arch:AVX)
	elseif(ARCH STREQUAL "arm")
		target_compile_options(compugraph PRIVATE /arch:ARMv7VE)
	endif()
else()
	target_compile_options(compugraph PRIVATE -march=native)
endif()