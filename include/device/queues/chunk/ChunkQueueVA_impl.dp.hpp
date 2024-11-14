#pragma once
#include "Parameters.h"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ChunkQueueVA.dp.hpp"
#include "device/ChunkAccess_impl.dp.hpp"
#include "device/BulkSemaphore_impl.dp.hpp"
#include "device/Chunk.dp.hpp"
#include "device/MemoryIndex.dp.hpp"
#include "device/queues/QueueChunk_impl.dp.hpp"

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ void
ChunkQueueVA<CHUNK_TYPE>::init(const Desc& d,MemoryManagerType *memory_manager)
{
        for (int i = d.item.get_group(0) * d.item.get_local_range(0) +
                     d.item.get_local_id(0);
             i < size_;
             i +=d. item.get_local_range(0) * d.item.get_group_range(0))
        {
		queue_[i] = DeletionMarker<index_t>::val;
	}

        if ((d.item.get_group(0) * d.item.get_local_range(0) +
             d.item.get_local_id(0)) == 0)
        {
		// Allocate 1 chunk per queue in the beginning
		index_t chunk_index{0};
		memory_manager->template allocateChunk<true>(chunk_index);
		auto chunk = QueueChunkType::initializeChunk(memory_manager->d_data, chunk_index, 0);
		queue_[0] = chunk_index;
	}
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ bool ChunkQueueVA<CHUNK_TYPE>::enqueueChunk(const Desc& d,
    MemoryManagerType *memory_manager, index_t chunk_index,
    index_t pages_per_chunk, typename MemoryManagerType::ChunkType *chunk)
{
        if (dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &count_, 1) < static_cast<int>(num_spots_))
        {
          enqueue(d,memory_manager, chunk_index, chunk);
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

	if (!FINAL_RELEASE)
                /*
                DPCT1015:1: Output needs adjustment.
                */
          d.out << "Queue "<<queue_index_<<": We died in EnqueueChunk with count "<<count_<<sycl::endl;
        assert(0); // no space to enqueue -> fail
        return false;
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ bool ChunkQueueVA<CHUNK_TYPE>::enqueueInitialChunk(
    MemoryManagerType *memory_manager, index_t chunk_index, int available_pages,
    index_t pages_per_chunk)
{
	// Increase count, insert chunk into queue
	++count_;
	++back_;
	auto queue_chunk = accessQueueElement(memory_manager, 0, 0);
	queue_chunk->enqueueInitial(0, chunk_index);
	semaphore.signal(available_pages);

	// Allocate one additional queue chunk
	index_t new_chunk_index{ 0 };
	memory_manager->allocateChunk<true>(new_chunk_index);
	QueueChunkType::initializeChunk(memory_manager->d_data, new_chunk_index, QueueChunkType::num_spots_);
	queue_[1] = new_chunk_index;
	return true;
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ void *
ChunkQueueVA<CHUNK_TYPE>::allocPage(const Desc& d,MemoryManagerType *memory_manager)
{
	using ChunkType = typename MemoryManagerType::ChunkType;

	uint32_t page_index, chunk_index;
	auto pages_per_chunk = MemoryManagerType::QI::getPagesPerChunkFromQueueIndex(queue_index_);
	ChunkType* chunk{ nullptr };

	semaphore.wait(d, 1, pages_per_chunk, [&]()
	{
		if (!memory_manager->template allocateChunk<false>(chunk_index))
		{
			if(!FINAL_RELEASE)
                          d.out<<"TODO: Could not allocate chunk!!!\n";
		}

		chunk = ChunkType::initializeChunk(memory_manager->d_data, chunk_index, pages_per_chunk, pages_per_chunk);
		// Please do NOT reorder here
                sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::device);
		enqueueChunk(d,memory_manager, chunk_index, pages_per_chunk, chunk);
	});

        /*
        DPCT1078:2: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

        unsigned int virtual_pos = Ouro::ldg_cg(&front_);
	while (true)
	{
		auto chunk_id = computeChunkID(virtual_pos);

		index_t queue_chunk_index{0};
		if((queue_chunk_index = Ouro::ldg_cg(&queue_[chunk_id])) == DeletionMarker<index_t>::val) 
		{
			++virtual_pos;
			continue;
		}

		auto queue_chunk = QueueChunkType::getAccess(memory_manager->d_data, queue_chunk_index);
		if(!queue_chunk->checkVirtualStart(virtual_pos))
		{
			if (!FINAL_RELEASE)
                          d.out<<"Virtualized does not match for chunk: "<<queue_chunk_index<<
                            " at position: "<<chunk_id<<
                            " with virtual start: "<<queue_chunk->virtual_start_<<
                            " ||| v_pos: "<<virtual_pos<<" || numspots "<<QueueChunkType::num_spots_<<sycl::endl;
                        assert(0);
                }

		queue_chunk->access(Ouro::modPower2<QueueChunkType::num_spots_>(virtual_pos), chunk_index);
		if (chunk_index != DeletionMarker<index_t>::val)
		{
			chunk = ChunkType::getAccess(memory_manager->d_data, chunk_index);
			const auto mode = chunk->access.allocPage(d,page_index);
			if (mode == ChunkType::ChunkAccessType::Mode::SUCCESSFULL)
				break;
			if (mode == ChunkType::ChunkAccessType::Mode::RE_ENQUEUE_CHUNK)
			{
				// Pretty special case, but we simply enqueue in the end again
                                if (dpct::atomic_fetch_add<
                                        sycl::access::address_space::
                                            generic_space>(&count_, 1) <
                                    static_cast<int>(num_spots_))
                                {
                                  enqueue(d,memory_manager, chunk_index, chunk);
				}
				break;
			}
			if (mode == ChunkType::ChunkAccessType::Mode::DEQUEUE_CHUNK)
			{
                                dpct::atomic_fetch_max<
                                    sycl::access::address_space::generic_space>(
                                    &front_, virtual_pos + 1);

                                // We moved the front pointer
				if(queue_chunk->deleteElement(Ouro::modPower2<QueueChunkType::num_spots_>(virtual_pos)))
				{
					// We can remove this chunk
					index_t reusable_chunk_id = atomicExch(&queue_[Ouro::modPower2<size_>(chunk_id)] , DeletionMarker<index_t>::val);
					// if(printDebug)
					// 	printf("We can reuse this chunk: %5u at position: %5u with virtual start: %10u | AllocPage-Reuse\n", reusable_chunk_id, virtual_pos, queue_chunk->virtual_start_);
					queue_chunk->cleanChunk();
					memory_manager->template enqueueChunkForReuse<false>(reusable_chunk_id);
				}
				// Reduce count again
                                dpct::atomic_fetch_sub<
                                    sycl::access::address_space::generic_space>(
                                    &count_, 1);

                                // if (atomicCAS(&front_, virtual_pos, virtual_pos + 1) == virtual_pos)
				// {
				// 	//printf("We can dequeue this chunk %u\n", chunk_id);
				// 	// Reduce count again
				// 	atomicSub(&count_, 1);

				// 	// We moved the front pointer
				// 	if(queue_chunk->deleteElement(Ouro::modPower2<QueueChunkType::num_spots_>(virtual_pos)))
				// 	{
				// 		// // We can remove this chunk
				// 		// index_t reusable_chunk_id = atomicExch(queue_ + chunk_id, DeletionMarker<index_t>::val);
				// 		// if(!FINAL_RELEASE && printDebug)
				// 		// 	printf("We can reuse this chunk: %5u at position: %5u with virtual start: %10u | AllocPage-Reuse\n", reusable_chunk_id, chunk_id, queue_chunk->virtual_start_);
				// 		// memory_manager->template enqueueChunkForReuse<false>(reusable_chunk_id);
				// 	}
				// }
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
                                  d.out<<"ThreadIDx: "<<d.item.get_local_id(0)<<" BlockIdx: "<<d.item.get_group(0)<<" - "
                                    "We done fucked up! Front: "<<virtual_pos<<
                                    "Back: "<<back_<<" : Count: "<<count_<<sycl::endl;
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
ChunkQueueVA<CHUNK_TYPE>::freePage(const Desc& d,MemoryManagerType *memory_manager,
                                   MemoryIndex index)
{
	using ChunkType = typename MemoryManagerType::ChunkType;

	auto chunk = ChunkType::getAccess(memory_manager->d_data, index.getChunkIndex());
	auto mode = chunk->access.freePage(d,index.getPageIndex());
	if(mode == ChunkType::ChunkAccessType::FreeMode::FIRST_FREE)
	{
		// Please do NOT reorder here
                /*
                DPCT1078:4: Consider replacing memory_order::acq_rel with
                memory_order::seq_cst for correctness if strong memory order
                restrictions are needed.
                */
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

                // We are the first to free something in this chunk, add it back to the queue
                if (dpct::atomic_fetch_add<
                        sycl::access::address_space::generic_space>(
                        &count_, 1) < static_cast<int>(num_spots_))
                {
                  enqueue(d,memory_manager, index.getChunkIndex(), chunk);
		}
		else
		{
			if (!FINAL_RELEASE)
                          d.out<<"Queue "<<queue_index_<<": We died in FreePage with count "<<count_<<sycl::endl;
                        assert(0);
                }
	}
	else if(mode == ChunkType::ChunkAccessType::FreeMode::DEQUEUE)
	{
		// TODO: Implement dequeue chunks
		auto num_pages_per_chunk{chunk->access.size};
		// Try to reduce semaphore
		if(semaphore.tryReduce(num_pages_per_chunk - 1))
		{
			// Lets try to flash the chunk
			if(false && chunk->access.tryFlashChunk())
			{
				// auto queue_chunk = accessQueueElement(memory_manager, index.getChunkIndex(), chunk->queue_pos);
				// if(queue_chunk->deleteElement(Ouro::modPower2<QueueChunkType::num_spots_>(chunk->queue_pos)))
				// {
				// 	// We can remove this chunk
				// 	index_t reusable_chunk_id = atomicExch(&queue_[Ouro::modPower2<size_>(computeChunkID(chunk->queue_pos))] , DeletionMarker<index_t>::val);
				// 	// if(printDebug)
				// 	// 	printf("We can reuse this chunk: %5u at position: %5u with virtual start: %10u | AllocPage-Reuse\n", reusable_chunk_id, virtual_pos, queue_chunk->virtual_start_);
				// 	queue_chunk->cleanChunk();
				// 	memory_manager->template enqueueChunkForReuse<false>(reusable_chunk_id);
				// }
				chunk->cleanChunk(reinterpret_cast<unsigned int*>(reinterpret_cast<memory_t*>(chunk) + ChunkType::size_));
				// atomicSub(&count_, 1);
				// memory_manager->enqueueChunkForReuse<false>(index.getChunkIndex());
				if(!FINAL_RELEASE && printDebug)
                                  d.out<<"Successfull re-use of chunk "<<index.getChunkIndex()<<sycl::endl;;
				if(statistics_enabled)
					atomicAdd(&(memory_manager->stats.chunkReuseCount), 1);
			}
			else
			{
				if(!FINAL_RELEASE && printDebug)
                                  d.out<<"Try Flash Chunk did not work!\n";
				// Flashing did not work, increase semaphore again by all pages
				semaphore.signal(num_pages_per_chunk);
			}
			return;
		}
		else
		{
			if(!FINAL_RELEASE && printDebug)
                          d.out<<"Try Reduce did not work for chunk "<<index.getChunkIndex()<<sycl::endl;
		}
	}

	// Please do NOT reorder here
        /*
        DPCT1078:3: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

        // Signal a free page
	semaphore.signal(1);
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ void ChunkQueueVA<CHUNK_TYPE>::enqueue(const Desc& d,
    MemoryManagerType *memory_manager, index_t index,
    typename MemoryManagerType::ChunkType *chunkindex_chunk)
{
        const unsigned int virtual_pos =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &back_, 1);
        auto chunk_id = computeChunkID(virtual_pos);
	const auto position = Ouro::modPower2<QueueChunkType::num_spots_>(virtual_pos);

	if (position == 0)
	{
		unsigned int chunk_index{ 0 };
		// We pre-emptively allocate the next chunk already
		memory_manager->template allocateChunk<true>(chunk_index);
		auto queue_chunk = QueueChunkType::initializeChunk(memory_manager->d_data, chunk_index, virtual_pos + QueueChunkType::num_spots_);
                /*
                DPCT1078:6: Consider replacing memory_order::acq_rel with
                memory_order::seq_cst for correctness if strong memory order
                restrictions are needed.
                */
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

                atomicExch(&queue_[Ouro::modPower2<size_>(chunk_id + 1)], chunk_index); 
	}

        /*
        DPCT1078:5: Consider replacing memory_order::acq_rel with
        memory_order::seq_cst for correctness if strong memory order
        restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

        auto chunk = accessQueueElement(d,memory_manager, chunk_id, virtual_pos);
	chunkindex_chunk->queue_pos = virtual_pos;
	if(QueueChunkType::checkChunkEmptyEnqueue(chunk->enqueue(d,position, index)))
	{
		// We can remove this chunk
		index_t reusable_chunk_id = atomicExch(queue_ + chunk_id, DeletionMarker<index_t>::val);
	  		Ouro::sleep();
		if(!FINAL_RELEASE && printDebug)
                  d.out<<"We can reuse this chunk: "<<reusable_chunk_id<<
                    " at position: "<<chunk_id<<
                    " with virtual start: "<<chunk->virtual_start_<<" | ENQUEUE-Reuse\n";
		memory_manager->template enqueueChunkForReuse<true>(reusable_chunk_id);
	}
}

// ##############################################################################################################################################
//
template <typename CHUNK_TYPE>
template <typename MemoryManagerType>
__dpct_inline__ QueueChunk<typename CHUNK_TYPE::Base> *
ChunkQueueVA<CHUNK_TYPE>::accessQueueElement(const Desc& d,MemoryManagerType *memory_manager,
                                             index_t chunk_id,
                                             index_t v_position)
{
	index_t queue_chunk_index{0};
	// We may have to wait until the first thread on this chunk has initialized it!
	unsigned int counter = 0;
	while((queue_chunk_index = Ouro::ldg_cg(&queue_[chunk_id])) == DeletionMarker<index_t>::val) 
	{
		Ouro::sleep(counter++);
	}

	auto queue_chunk = QueueChunkType::getAccess(memory_manager->d_data, queue_chunk_index);
	if(!queue_chunk->checkVirtualStart(v_position))
	{
		if (!FINAL_RELEASE)
                  d.out<<"Virtualized does not match for chunk: "<<queue_chunk_index<<
                    " at position: "<<chunk_id<<
                    " with virtual start: "<<queue_chunk->virtual_start_<<
                    " ||| v_pos: "<<v_position<<
                    " || numspots "<<QueueChunkType::num_spots_<<sycl::endl;
                assert(0);
        }

	return queue_chunk;
}
