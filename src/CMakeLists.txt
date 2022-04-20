include_directories(${SIMPLEDNN_HEADER_DIR})
FILE(GLOB SIMPLEDNN_SOURCES ./**.cpp)
add_library(simplednn ${SIMPLEDNN_SOURCES})
target_link_libraries(simplednn ${lapackblas_libraries})