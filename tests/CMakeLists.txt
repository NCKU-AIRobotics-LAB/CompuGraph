include_directories(${COMPUGRAPH_HEADER_DIR})
FILE(GLOB TEST_SOURCES ./**.cpp)
add_executable(test ${TEST_SOURCES})
target_link_libraries(test cg)

# Put this code after every executable target, if is Python library, setup.py will also copy the file to the destination.
if(NEED_COPY_OPENBLAS)
	add_custom_command(
		TARGET test POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy 
			${EXTERNAL_INSTALL_LOCATION}/bin/libopenblas.dll
			$<TARGET_FILE_DIR:test>
	)
endif()

# for xsimd
if(MSVC)
	set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
	if(HAVE_AVX2_EXTENSIONS AND NOT MSVC_VERSION LESS 1800)
		target_compile_options(test PRIVATE /arch:AVX2)
	elseif(HAVE_AVX_EXTENSIONS AND NOT MSVC_VERSION LESS 1600)
		target_compile_options(test PRIVATE /arch:AVX)
	elseif(ARCH STREQUAL "arm")
		target_compile_options(test PRIVATE /arch:ARMv7VE)
	endif()
else()
	target_compile_options(test PRIVATE -march=native)
endif()