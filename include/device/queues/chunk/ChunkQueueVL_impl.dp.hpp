#pragma once
#include "Parameters.h"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ChunkQueueVL.dp.hpp"
#include "device/ChunkAccess_impl.cuh"
#include "device/BulkSemaphore_impl.cuh"
#include "device/Chunk.cuh"
#include "device/MemoryIndex.cuh"
#include "device/queues/QueueChunk_impl.cuh"

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ void
ChunkQueueVL<CHUNK_TYPE>::init(MemoryManagerType *memory_manager,
                               const sycl::nd_item<3> &item_ct1)
{
        if ((item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2)) == 0)
        {
		// Allocate 1 chunk per queue in the beginning
		index_t chunk_index{0};
		memory_manager->allocateChunk<true>(chunk_index);
		auto queue_chunk = QueueChunkType::initializeChunk(memory_manager->d_data, chunk_index, 0);

		if(!FINAL_RELEASE && printDebug)
			printf("Allocate a new chunk for the queue %u with index: %u : ptr: %p\n",queue_index_, chunk_index, queue_chunk);
		
		// All pointers point to the same chunk in the beginning
		front_ptr_ = queue_chunk;
		back_ptr_ = queue_chunk;
		old_ptr_ = queue_chunk;
	}
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ bool
ChunkQueueVL<CHUNK_TYPE>::enqueueChunk(MemoryManagerType *memory_manager,
                                       index_t chunk_index,
                                       index_t pages_per_chunk)
{
	enqueue(memory_manager, chunk_index);

	// Please do NOT reorder here
        /*
        DPCT1078:0: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

        semaphore.signalExpected(pages_per_chunk);
	return true;
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ bool ChunkQueueVL<CHUNK_TYPE>::enqueueInitialChunk(
    MemoryManagerType *memory_manager, index_t chunk_index, int available_pages,
    index_t pages_per_chunk)
{
	enqueue(memory_manager, chunk_index);
	semaphore.signal(available_pages);
	return true;
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ void *
ChunkQueueVL<CHUNK_TYPE>::allocPage(MemoryManagerType *memory_manager,
                                    const sycl::nd_item<3> &item_ct1)
{
	using ChunkType = typename MemoryManagerType::ChunkType;

	MemoryIndex index;
	uint32_t chunk_index, page_index;
	auto pages_per_chunk = MemoryManagerType::QI::getPagesPerChunkFromQueueIndex(queue_index_);
	ChunkType* chunk{ nullptr };

	semaphore.wait(1, pages_per_chunk, [&]()
	{
		if (!memory_manager->allocateChunk<false>(chunk_index))
		{
			if(!FINAL_RELEASE)
				printf("TODO: Could not allocate chunk!!!\n");
		}

		ChunkType::initializeChunk(memory_manager->d_data, chunk_index, pages_per_chunk, pages_per_chunk);
		__threadfence();
		enqueueChunk(memory_manager, chunk_index, pages_per_chunk);
		__threadfence();
	});

        /*
        DPCT1078:1: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::work_group);

        unsigned int virtual_pos = Ouro::ldg_cg(&front_);
	auto queue_chunk = front_ptr_;
	while(true)
	{
		queue_chunk = queue_chunk->accessLinked(virtual_pos);

                /*
                DPCT1078:2: Consider replacing memory_order::acq_rel with
                memory_order::seq_cst for correctness if strong memory order
                restrictions are needed.
                */
                sycl::atomic_fence(sycl::memory_order::acq_rel,
                                   sycl::memory_scope::work_group);

                // This position might be out-dated already
		queue_chunk->access(Ouro::modPower2<QueueChunkType::num_spots_>(virtual_pos), chunk_index);
		if(chunk_index != DeletionMarker<index_t>::val)
		{
			chunk = ChunkType::getAccess(memory_manager->d_data, chunk_index);
			const auto mode = chunk->access.allocPage(page_index);
			
			if (mode == ChunkType::ChunkAccessType::Mode::SUCCESSFULL)
				break;
			if (mode == ChunkType::ChunkAccessType::Mode::RE_ENQUEUE_CHUNK)
			{
				// Pretty special case, but we simply enqueue in the end again
				enqueue(memory_manager, chunk_index);
				break;
			}
			if (mode == ChunkType::ChunkAccessType::Mode::DEQUEUE_CHUNK)
			{
				// if (atomicCAS(&front_, virtual_pos, virtual_pos + 1) == virtual_pos)
				// {
				// 	front_ptr_->dequeue<QueueChunkType::DEQUEUE_MODE::DELETE>(memory_manager, virtual_pos, index.index, &front_ptr_, &old_ptr_, &old_count_);
				// }

				// TODO: Why does this not work
                                dpct::atomic_fetch_max<
                                    sycl::access::address_space::generic_space>(
                                    &front_, virtual_pos + 1);
                                front_ptr_->dequeue<QueueChunkType::DEQUEUE_MODE::DELETE>(memory_manager, virtual_pos, index.index, &front_ptr_, &old_ptr_, &old_count_);
				break;
			}
		}

		// Check next chunk
		++virtual_pos;
		// ##############################################################################################################
		// Error Checking
		if (!FINAL_RELEASE)
		{
			if (virtual_pos > Ouro::ldg_cg(&back_))
			{
				if (!FINAL_RELEASE)
                                        printf("ThreadIDx: %d BlockIdx: %d - "
                                               "Front: %u Back: %u - "
                                               "ChunkIndex: %u\n",
                                               item_ct1.get_local_id(2),
                                               item_ct1.get_group(2),
                                               virtual_pos, back_, chunk_index);
                                assert(0);
                        }
		}
	}

	return ChunkType::getPage(memory_manager->d_data, chunk_index, page_index, page_size_);
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ void
ChunkQueueVL<CHUNK_TYPE>::freePage(MemoryManagerType *memory_manager,
                                   MemoryIndex index)
{
	using ChunkType = typename MemoryManagerType::ChunkType;

	uint32_t chunk_index, page_index;
	index.getIndex(chunk_index, page_index);
	auto chunk = ChunkType::getAccess(memory_manager->d_data, index.getChunkIndex());
	auto mode = chunk->access.freePage(index.getPageIndex());
	if(mode == ChunkType::ChunkAccessType::FreeMode::FIRST_FREE)
	{
		enqueue(memory_manager, index.getChunkIndex());
	}
	else if(mode == ChunkType::ChunkAccessType::FreeMode::DEQUEUE)
	{
		// TODO: Implement dequeue chunks
		// auto num_pages_per_chunk{chunk->access.size};
		// // Try to reduce semaphore
		// if(semaphore.tryReduce(num_pages_per_chunk - 1))
		// {
		// 	// Lets try to flash the chunk
		// 	if(false && chunk->access.tryFlashChunk())
		// 	{
		// 		chunk->cleanChunk(reinterpret_cast<unsigned int*>(reinterpret_cast<memory_t*>(chunk) + ChunkType::size_));
		// 		front_ptr_->dequeue<QueueChunkType::DEQUEUE_MODE::DELETE>(memory_manager, chunk->queue_pos, index.index, &front_ptr_, &old_ptr_, &old_count_);
		// 		memory_manager->enqueueChunkForReuse<false>(index.getChunkIndex());
		// 		if(!FINAL_RELEASE && printDebug)
		// 			printf("Successfull re-use of chunk %u\n", index.getChunkIndex());
		// 		if(statistics_enabled)
		// 			atomicAdd(&(memory_manager->stats.chunkReuseCount), 1);
		// 	}
		// 	else
		// 	{
		// 		if(!FINAL_RELEASE && printDebug)
		// 			printf("Try Flash Chunk did not work!\n");
		// 		// Flashing did not work, increase semaphore again by all pages
		// 		semaphore.signal(num_pages_per_chunk);
		// 	}
		// 	return;
		// }
		// else
		// {
		// 	if(!FINAL_RELEASE && printDebug)
		// 		printf("Try Reduce did not work for chunk %u!\n", index.getChunkIndex());
		// }
	}
	// Please do NOT reorder here
        /*
        DPCT1078:3: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::work_group);

        // Signal a free page
	semaphore.signal(1);
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ void
ChunkQueueVL<CHUNK_TYPE>::enqueue(MemoryManagerType *memory_manager,
                                  index_t index)
{
	// Increase back and compute the position on a chunk
        const unsigned int virtual_pos =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &back_, 1);
        back_ptr_->enqueue(memory_manager, virtual_pos, index, &back_ptr_, &front_ptr_, &old_ptr_, &old_count_);
}