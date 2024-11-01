#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once
#include "device/MemoryIndex.dp.hpp"
#include "device/queues/chunk/ChunkQueue.dp.hpp"
#include "device/ChunkAccess_impl.dp.hpp"
#include "device/BulkSemaphore_impl.dp.hpp"
#include "device/Chunk.dp.hpp"

// ##############################################################################################################################################
//
template <typename ChunkType>
template <typename MemoryManagerType>
__dpct_inline__ void
ChunkQueue<ChunkType>::init(MemoryManagerType *memory_manager, sycl::nd_item<1> item)
{
  //for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size_; i += blockDim.x * gridDim.x)
  for (int i = item.get_global_id(0); i < size_; i += item.get_global_range(0))
	{
		queue_[i] = DeletionMarker<index_t>::val;
	}
}

// ##############################################################################################################################################
//
template <typename ChunkType>
__dpct_inline__ bool ChunkQueue<ChunkType>::enqueue(index_t chunk_index,
                                                    ChunkType *chunk)
{
  //int fill = atomicAdd(&count_, 1);
  int fill=++atomicCount;
	if (fill < static_cast<int>(size_))
	{
		// we have to wait in case there is still something in the spot
		// note: as the filllevel could be increased by this thread, we are certain that the spot will become available
		//unsigned int pos = Ouro::modPower2<size_>(atomicAdd(&back_, 1));
          unsigned int pos = Ouro::modPower2<size_>(atomicBack++);

		// unsigned int counter{0};
          while (Ouro::atomicCAS(queue_[pos], DeletionMarker<index_t>::val, chunk_index) != DeletionMarker<index_t>::val)
		{
			Ouro::sleep();
			// Ouro::sleep(++counter);
			// if(++counter > 1000)
			// {
			// 	printf("%d - %d Died in ChunkQueue::enqueue!\n", threadIdx.x, blockIdx.x);
			// }
		}

          //__threadfence_block();
                sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::work_group);

		if(chunk)
			atomicExch(&chunk->queue_pos, pos);
		return true;
	}

//	if(!FINAL_RELEASE)
//		printf("This chunks is most likely lost for now\n");
//	__trap(); //no space to enqueue -> fail // TODO
	return false;
}

// ##############################################################################################################################################
//
template <typename ChunkType>
__dpct_inline__ bool
ChunkQueue<ChunkType>::enqueueChunk(index_t chunk_index,
                                    index_t pages_per_chunk, ChunkType *chunk)
{
	const auto retVal = enqueue(chunk_index, chunk);
	// Please do NOT reorder here
	//__threadfence_block();
        sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::work_group);

	semaphore.signalExpected(pages_per_chunk);
	return retVal;
}

// ##############################################################################################################################################
//
template <typename ChunkType>
template <typename MemoryManagerType>
__dpct_inline__ bool ChunkQueue<ChunkType>::enqueueInitialChunk(
    MemoryManagerType *memory_manager, index_t chunk_index, int availablePages,
    index_t pages_per_chunk)
{
	count_ = 1;
	queue_[back_++] = chunk_index;
	semaphore.signal(availablePages);
	return true;
}

// ##############################################################################################################################################
//
template <typename ChunkType>
template <typename MemoryManagerType>
__dpct_inline__ void *
ChunkQueue<ChunkType>::allocPage(MemoryManagerType *memory_manager,const sycl::nd_item<1>& item)
{
	uint32_t page_index, chunk_index;
	auto pages_per_chunk = MemoryManagerType::QI::getPagesPerChunkFromQueueIndex(queue_index_);
	ChunkType* chunk{ nullptr };

	// Try to allocate a chunk
	semaphore.wait(1, pages_per_chunk, [&]()
	{
		if (!memory_manager->allocateChunk<false>(chunk_index))
		{
			if (!FINAL_RELEASE)
				printf("TODO: Could not allocate chunk!!!\n");
		}
	 	chunk = ChunkType::initializeEmptyChunk(memory_manager->d_data, chunk_index, pages_per_chunk);
		 // Please do NOT reorder here
		//__threadfence_block();
                sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::work_group);
                enqueueChunk(chunk_index, pages_per_chunk, chunk);
	});

	auto current_front = Ouro::ldg_cg(&front_);
	while(true)
	{
		chunk_index = Ouro::ldg_cg(&queue_[Ouro::modPower2<size_>(current_front)]);
		if(chunk_index != DeletionMarker<index_t>::val)
		{
			chunk = ChunkType::getAccess(memory_manager->d_data, chunk_index);
			const auto mode = chunk->access.allocPage(page_index,item);
			if (mode == ChunkType::ChunkAccessType::Mode::SUCCESSFULL)
				break;

			if(mode == ChunkType::ChunkAccessType::Mode::RE_ENQUEUE_CHUNK)
			{
				// Pretty special case, but we simply enqueue in the end again
				enqueue(chunk_index, chunk);
				break;
			}
			if (mode == ChunkType::ChunkAccessType::Mode::DEQUEUE_CHUNK)
			{
				// We moved the front pointer
                          //atomicMax(&front_, current_front + 1);
                          atomicFront.fetch_max(current_front + 1);
                          //atomicExch(&chunk->queue_pos, DeletionMarker<index_t>::val);
                          Ouro::Atomic<unsigned>(chunk->queue_pos)=DeletionMarker<index_t>::val;
                          //atomicExch(&queue_[Ouro::modPower2<size_>(current_front)], DeletionMarker<index_t>::val);
                          Ouro::Atomic<unsigned>{queue_[Ouro::modPower2<size_>(current_front)]}=DeletionMarker<index_t>::val;
                          //atomicSub(&count_, 1);
                          atomicCount--;
                          break;
			}
		}
		
		// Check next chunk
		++current_front;

		// ##############################################################################################################
		// Error Checking
		if (!FINAL_RELEASE)
		{
			if (current_front > Ouro::ldg_cg(&back_))
			{
                          // TODO
                          //printf("ThreadIDx: %d BlockIdx: %d - We done fucked up! Front: %u Back: %u : Count: %d\n", threadIdx.x, blockIdx.x, current_front, back_, count_);
                          //__trap();
			}
		}
	}
	
	return chunk->getPage(memory_manager->d_data, chunk_index, page_index, page_size_);
}

// ##############################################################################################################################################
//
template <typename ChunkType>
template <typename MemoryManagerType>
__dpct_inline__ void
ChunkQueue<ChunkType>::freePage(MemoryManagerType *memory_manager,
                                MemoryIndex index)
{
	auto chunk = ChunkType::getAccess(memory_manager->d_data, index.getChunkIndex());
	auto mode = chunk->access.freePage(index.getPageIndex());
	if(mode == ChunkType::ChunkAccessType::FreeMode::FIRST_FREE)
	{
		// We are the first to free something in this chunk, add it back to the queue
		enqueue(index.getChunkIndex(), chunk);
	}
	else if(mode == ChunkType::ChunkAccessType::FreeMode::DEQUEUE && Ouro::ldg_cg(&count_) > lower_fill_level)
	{
		auto num_pages_per_chunk{chunk->access.size};

		// Try to reduce semaphore
		if(semaphore.tryReduce(num_pages_per_chunk - 1))
		{
			// Lets try to flash the chunk
			if(chunk->access.tryFlashChunk())
			{
				// Make sure that a previous enqueue operation is finished -> we have a valid queue position
				while(atomicCAS(&chunk->queue_pos, DeletionMarker<index_t>::val, DeletionMarker<index_t>::val) == DeletionMarker<index_t>::val)
					Ouro::sleep();
				// Reduce queue count, take element out of queue and put it into the reuse queue
                                //atomicExch(queue_ + chunk->queue_pos, DeletionMarker<index_t>::val);
                                Ouro::Atomic<unsigned>{queue_[chunk->queue_pos]}=DeletionMarker<index_t>::val;
				//atomicSub(&count_, 1);
                                atomicCount--;
				memory_manager->enqueueChunkForReuse<false>(index.getChunkIndex());
				if(!FINAL_RELEASE && printDebug)
					printf("Successfull re-use of chunk %u\n", index.getChunkIndex());
				if(statistics_enabled)
					atomicAdd(&(memory_manager->stats.chunkReuseCount), 1);
			}
			else
			{
				if(!FINAL_RELEASE && printDebug)
					printf("Try Flash Chunk did not work!\n");
				// Flashing did not work, increase semaphore again by all pages
				semaphore.signal(num_pages_per_chunk);
			}
			return;
		}
		else
		{
			if(!FINAL_RELEASE && printDebug)
				printf("Try Reduce did not work for chunk %u!\n", index.getChunkIndex());
		}
	}

	// Please do NOT reorder here
	//__threadfence_block();
        sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::work_group);

	// Signal a free page
	semaphore.signal(1);
}
