#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once
#include "device/MemoryIndex.dp.hpp"
#include "device/queues/chunk/ChunkQueue.dp.hpp"
#include "device/ChunkAccess_impl.dp.hpp"
#include "device/BulkSemaphore_impl.dp.hpp"
#include "device/Chunk.dp.hpp"

namespace Ouro
{
  // ##############################################################################################################################################
  //
  template <typename ChunkType>
  template <typename Desc,typename MemoryManagerType>
  __dpct_inline__ void
  ChunkQueue<ChunkType>::init(const Desc& d,MemoryManagerType *memory_manager)
  {
    for (int i = d.item.get_global_linear_id(); i < size_; i += d.item.get_global_range().size())
      {
        queue_[i] = DeletionMarker<index_t>::val;
      }
  }

  // ##############################################################################################################################################
  //
  template <typename ChunkType>
  template <typename Desc>
  __dpct_inline__ bool ChunkQueue<ChunkType>::enqueue(const Desc& d,index_t chunk_index,
                                                      ChunkType *chunk)
  {
    int fill = atomicAdd(&count_, 1);
    if (fill < static_cast<int>(size_))
      {
        // we have to wait in case there is still something in the spot
        // note: as the filllevel could be increased by this thread, we are certain that the spot will become available
        unsigned int pos = Ouro::modPower2<size_>(atomicAdd(&back_, 1));

        // unsigned int counter{0};
        while (atomicCAS(queue_ + pos, DeletionMarker<index_t>::val, chunk_index) != DeletionMarker<index_t>::val)
          {
            Ouro::sleep();
          }

        //__threadfence_block();
        sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::work_group);

        if(chunk)
          atomicExch(&chunk->queue_pos, pos);
        return true;
      }

    if(!FINAL_RELEASE)
      d.out<<"This chunks is most likely lost for now\n";
    return false;
  }

  // ##############################################################################################################################################
  //
  template <typename ChunkType>
  template <typename Desc>
  __dpct_inline__ bool
  ChunkQueue<ChunkType>::enqueueChunk(const Desc& d,index_t chunk_index,
                                      index_t pages_per_chunk, ChunkType *chunk)
  {
    const auto retVal = enqueue(d,chunk_index, chunk);
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
  template <typename Desc,typename MemoryManagerType>
  __dpct_inline__ void *
  ChunkQueue<ChunkType>::allocPage(const Desc& d,MemoryManagerType *memory_manager)
  {
    uint32_t page_index, chunk_index;
    auto pages_per_chunk = MemoryManagerType::QI::getPagesPerChunkFromQueueIndex(queue_index_);
    ChunkType* chunk{ nullptr };

    // Try to allocate a chunk
    semaphore.wait(d,1, pages_per_chunk, [&]()
    {
      if (!memory_manager->template allocateChunk<false>(chunk_index))
        {
          if(!FINAL_RELEASE)
            d.out<<"TODO: Could not allocate chunk!!!\n";
        }
      chunk = ChunkType::initializeEmptyChunk(memory_manager->d_data, chunk_index, pages_per_chunk);
      // Please do NOT reorder here
      //__threadfence_block();
      sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::work_group);
      enqueueChunk(d,chunk_index, pages_per_chunk, chunk);
    });

    auto current_front = front_;
    while(true)
      {
        chunk_index = queue_[Ouro::modPower2<size_>(current_front)];
        if(chunk_index != DeletionMarker<index_t>::val)
          {
            chunk = ChunkType::getAccess(memory_manager->d_data, chunk_index);
            const auto mode = chunk->access.allocPage(d,page_index);
            if (mode == ChunkType::ChunkAccessType::Mode::SUCCESSFULL)
              break;

            if(mode == ChunkType::ChunkAccessType::Mode::RE_ENQUEUE_CHUNK)
              {
                // Pretty special case, but we simply enqueue in the end again
                enqueue(d,chunk_index, chunk);
                break;
              }
            if (mode == ChunkType::ChunkAccessType::Mode::DEQUEUE_CHUNK)
              {
                // We moved the front pointer
                atomicMax(&front_, current_front + 1);
                atomicExch(&chunk->queue_pos, DeletionMarker<index_t>::val);
                atomicExch(&queue_[Ouro::modPower2<size_>(current_front)], DeletionMarker<index_t>::val);
                atomicSub(&count_, 1);
                break;
              }
          }
		
        // Check next chunk
        ++current_front;

        // ##############################################################################################################
        // Error Checking
        if (!FINAL_RELEASE)
          {
            if (current_front > back_)
              {
                d.out<<"ThreadIDx: "<<d.item.get_local_linear_id()<<" BlockIdx: "<<d.item.get_group_linear_id()<<" - We done fucked up! Front: "<<current_front<<" Back: "<<back_<<" : Count: "<<count_<<sycl::endl;
                return nullptr;
              }
          }
      }
	
    return chunk->getPage(memory_manager->d_data, chunk_index, page_index, page_size_);
  }

  // ##############################################################################################################################################
  //
  template <typename ChunkType>
  template <typename Desc,typename MemoryManagerType>
  __dpct_inline__ void
  ChunkQueue<ChunkType>::freePage(const Desc& d,MemoryManagerType *memory_manager,
                                  MemoryIndex index)
  {
    auto chunk = ChunkType::getAccess(memory_manager->d_data, index.getChunkIndex());
    auto mode = chunk->access.freePage(d,index.getPageIndex());
    if(mode == ChunkType::ChunkAccessType::FreeMode::FIRST_FREE)
      {
        // We are the first to free something in this chunk, add it back to the queue
        enqueue(d,index.getChunkIndex(), chunk);
      }
    else if(mode == ChunkType::ChunkAccessType::FreeMode::DEQUEUE && count_ > lower_fill_level)
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
                atomicExch(queue_ + chunk->queue_pos, DeletionMarker<index_t>::val);
                atomicSub(&count_, 1);
                memory_manager->template enqueueChunkForReuse<false>(index.getChunkIndex());
                if(!FINAL_RELEASE && printDebug)
                  d.out<<"Successfull re-use of chunk "<<index.getChunkIndex()<<sycl::endl;
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
    //__threadfence_block();
    sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::work_group);

    // Signal a free page
    semaphore.signal(1);
  }
}
