cmake_minimum_required(VERSION 3.10)
project(SIMPLEDNN)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SIMPLEDNN_HEADER_DIR ${SIMPLEDNN_SOURCE_DIR}/include)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} 
)
ExternalProject_Add(xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
		DEPENDS xtl
)
ExternalProject_Add(xtensor-blas
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
		DEPENDS xtensor
)
ExternalProject_Add(xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
		DEPENDS xtensor-blas
)
ExternalProject_Add(xtensor-python
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-python
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DPYBIND11_PYTHON_VERSION=${PYBIND11_PYTHON_VERSION}
		DEPENDS xsimd
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
include_directories(${SIMPLEDNN_HEADER_DIR})
FILE(GLOB PYBIND_SOURCES python/**.cpp)
find_package(Python3 ${PYBIND11_PYTHON_VERSION} COMPONENTS NumPy REQUIRED)
pybind11_add_module(pysimplednn ${PYBIND_SOURCES})
target_link_libraries(pysimplednn PUBLIC simplednn Python3::NumPy)