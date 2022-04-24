include_directories(${COMPUGRAPH_HEADER_DIR})
FILE(GLOB TEST_SOURCES ./**.cpp)
add_executable(test ${TEST_SOURCES})
target_link_libraries(test cg)