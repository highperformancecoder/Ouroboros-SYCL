#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once

#include "Definitions.h"
#include "Parameters.h"
#include "device/BulkSemaphore.dp.hpp"
#include "device/queues/QueueChunk.dp.hpp"

// Forward declaration
struct MemoryIndex;

template <typename CHUNK_TYPE>
struct PageQueueVL
{
	using SemaphoreType = BulkSemaphore;
	using ChunkType = CHUNK_TYPE;
	using Base = typename ChunkType::Base;
	using QueueChunkType = QueueChunk<Base>;

	// Members
	index_t* queue_{nullptr}; // Chunk-Pointers
	int queue_index_{ 0 }; // Which queue is this!
	int page_size_{ 0 };
	SemaphoreType semaphore{ SemaphoreType::null_value }; // Access Management
	unsigned int front_{ 0 }; // Current front (virtual)
	unsigned int back_{ 0 }; // Current back (virtual)
	QueueChunkType* front_ptr_{nullptr}; // Current front chunk
	QueueChunkType* back_ptr_{nullptr}; // Current back chunk
	QueueChunkType* old_ptr_{nullptr}; // Old chunk ptr
	unsigned int old_count_{0}; // How many old chunks are there?

	// Static Members
	static constexpr bool virtualized{true};
	static constexpr int size_{ 0 }; // How many chunk pointers can we store?
	static constexpr int lower_fill_level{static_cast<int>(static_cast<float>(virtual_queue_size * QueueChunkType::num_spots_) * LOWER_FILL_LEVEL_PERCENTAGE)};

	// Methods
        template <typename MemoryManagerType>
        __dpct_inline__ bool preFillQueue(MemoryManagerType *memory_manager,
                                          index_t chunk_index,
                                          index_t pages_per_chunk)
        {
		return enqueueChunk(memory_manager, chunk_index, pages_per_chunk);
	}

        __dpct_inline__ uint32_t getCount() const
        {
		return semaphore.getCount();
	}

        template <typename MemoryManagerType>
        __dpct_inline__ void init(const Desc&,MemoryManagerType *memory_manager);

        template <typename MemoryManagerType>
        __dpct_inline__ bool
        enqueueInitialChunk(MemoryManagerType *memory_manager,
                            index_t chunk_index, int available_pages,
                            index_t pages_per_chunk);

        template <typename MemoryManagerType>
        __dpct_inline__ void *allocPage(const Desc&,MemoryManagerType *memory_manager);

        template <typename MemoryManagerType>
        __dpct_inline__ void freePage(const Desc&,MemoryManagerType *memory_manager,
                                      MemoryIndex index);

        template <typename MemoryManagerType>
        __dpct_inline__ void enqueue(const Desc&, MemoryManagerType *memory_manager,
                                     index_t index);
        template <typename MemoryManagerType>
        __dpct_inline__ bool enqueueChunk(const Desc&,MemoryManagerType *memory_manager,
                                          index_t chunk_index,
                                          index_t pages_per_chunk);
};
