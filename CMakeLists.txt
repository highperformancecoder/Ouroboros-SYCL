cmake_minimum_required(VERSION 3.16)
# note - setting the compiler here doesn't appear to work. Run ccmake
# after cmake, and manually set the compiler name
#set(CMAKE_CXX_COMPILER,icpx)
#add_compile_options(-fsycl)
# also -fsycl-targets=nvptx64-nvidia-cuda on NVidia environments
#add_link_options(-fsycl)

project(ouroboros LANGUAGES CXX VERSION 1.0)

#include(BaseCMake.cmake)

option(TEST_P "Test Page-Based Standard" True)
option(TEST_VAP "Test Page-Based VA" True)
option(TEST_VLP "Test Page-Based VL" True)
option(TEST_C "Test Chunk-Based Standard" True)
option(TEST_VAC "Test Chunk-Based VA" True)
option(TEST_VLC "Test Chunk-Based VL" True)


##########################################################################
# Executables
##########################################################################

SET(OUROBOROSHEADERS
	# General Stuff
	include/Definitions.h
	include/Utility.h
	include/Utility.dp.hpp
	include/Parameters.h
	include/device/Helper.dp.hpp

	# Access Primitive
	include/device/BulkSemaphore.dp.hpp
	include/device/BulkSemaphore_impl.dp.hpp

	# Memory
	include/device/MemoryIndex.dp.hpp

	# Chunk Stuff
	include/device/Chunk.dp.hpp
	include/device/ChunkIndexChunk.dp.hpp
	include/device/ChunkAccess.dp.hpp
	include/device/ChunkAccess_impl.dp.hpp
	include/device/PageIndexChunk.dp.hpp
	include/device/queues/QueueChunk.dp.hpp
	include/device/queues/QueueChunk_impl.dp.hpp

	# Queues
	include/device/queues/Queue.h
	include/device/queues/Queue_impl.dp.hpp
	include/device/queues/Queues.dp.hpp
	include/device/queues/Queues_impl.dp.hpp
	include/device/queues/page/PageQueue.dp.hpp
	include/device/queues/page/PageQueue_impl.dp.hpp
	include/device/queues/page/PageQueueVA.dp.hpp
	include/device/queues/page/PageQueueVA_impl.dp.hpp
	include/device/queues/page/PageQueueVL.dp.hpp
	include/device/queues/page/PageQueueVL_impl.dp.hpp
	include/device/queues/chunk/ChunkQueue.dp.hpp
	include/device/queues/chunk/ChunkQueue_impl.dp.hpp
	include/device/queues/chunk/ChunkQueueVA.dp.hpp
	include/device/queues/chunk/ChunkQueueVA_impl.dp.hpp
	include/device/queues/chunk/ChunkQueueVL.dp.hpp
	include/device/queues/chunk/ChunkQueueVL_impl.dp.hpp

	# Ouroboros
	include/device/Ouroboros.dp.hpp
	include/device/Ouroboros_impl.dp.hpp
	include/device/MemoryInitialization.dp.hpp
)

if(TEST_P)
add_executable(main_p
	src/main.cpp
	${OUROBOROSHEADERS}
)
target_compile_definitions(main_p PUBLIC TEST_PAGES)
target_include_directories(main_p PUBLIC include)
endif()

if(TEST_C)
add_executable(main_c
	src/main.cpp
	${OUROBOROSHEADERS}
)
target_compile_definitions(main_c PUBLIC TEST_CHUNKS)
target_include_directories(main_c PUBLIC include)
endif()

if(TEST_VAP)
add_executable(va_main_p
	src/main.cpp
	${OUROBOROSHEADERS}
)
target_compile_definitions(va_main_p PUBLIC TEST_PAGES TEST_VIRTUALARRAY)
target_include_directories(va_main_p PUBLIC include)
endif()

if(TEST_VAC)
add_executable(va_main_c
	src/main.cpp
	${OUROBOROSHEADERS}
)
target_compile_definitions(va_main_c PUBLIC TEST_CHUNKS TEST_VIRTUALARRAY)
target_include_directories(va_main_c PUBLIC include)
endif()

if(TEST_VLP)
add_executable(vl_main_p
	src/main.cpp
	${OUROBOROSHEADERS}
)
target_compile_definitions(vl_main_p PUBLIC TEST_PAGES TEST_VIRTUALLIST)
target_include_directories(vl_main_p PUBLIC include)
endif()

if(TEST_VLC)
add_executable(vl_main_c
	src/main.cpp
	${OUROBOROSHEADERS}
)
target_compile_definitions(vl_main_c PUBLIC TEST_CHUNKS TEST_VIRTUALLIST)
target_include_directories(vl_main_c PUBLIC include)
endif()
