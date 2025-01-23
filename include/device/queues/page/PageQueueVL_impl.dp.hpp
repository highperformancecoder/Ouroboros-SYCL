#pragma once
#include "Parameters.h"
#include <sycl/sycl.hpp>
#include "PageQueueVL.dp.hpp"
#include "device/BulkSemaphore_impl.dp.hpp"
#include "device/Chunk.dp.hpp"
#include "device/MemoryIndex.dp.hpp"
#include "device/queues/QueueChunk_impl.dp.hpp"

namespace Ouro
{
  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename Desc,typename MemoryManagerType>
  inline void
  PageQueueVL<CHUNK_TYPE>::init(const Desc& d,MemoryManagerType *memory_manager)
  {
    if (d.item.get_global_linear_id() == 0)
      {
        // Allocate 1 chunk per queue in the beginning
        index_t chunk_index{0};
        memory_manager->template allocateChunk<true>(chunk_index);
        auto queue_chunk = QueueChunkType::initializeChunk(memory_manager->d_data, chunk_index, 0);

        if(!FINAL_RELEASE && printDebug)
          d.out<<"Allocate a new chunk for the queue "<<queue_index_<<" with index: "<<chunk_index<<" : ptr: "<<queue_chunk<<sycl::endl;
		
        // All pointers point to the same chunk in the beginning
        front_ptr_ = queue_chunk;
        back_ptr_ = queue_chunk;
        old_ptr_ = queue_chunk;
      }
  }

  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename Desc,typename MemoryManagerType>
  inline bool
  PageQueueVL<CHUNK_TYPE>::enqueueChunk(const Desc& d,MemoryManagerType *memory_manager,
                                        index_t chunk_index,
                                        index_t pages_per_chunk)
  {
    unsigned int virtual_pos = atomicAdd(&back_, pages_per_chunk);

    back_ptr_->enqueueChunk(d, memory_manager, virtual_pos, chunk_index, pages_per_chunk, &back_ptr_, &front_ptr_, &old_ptr_, &old_count_);

    // Please DO NOT reorder here
    sycl::atomic_fence(sycl::memory_order::seq_cst,
                       sycl::memory_scope::work_group);

    // Since our space is not limited, we can signal at the end
    semaphore.signalExpected(pages_per_chunk);
    return true;
  }

  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename MemoryManagerType>
  inline bool PageQueueVL<CHUNK_TYPE>::enqueueInitialChunk(
                                                                    MemoryManagerType *memory_manager, index_t chunk_index, int available_pages,
                                                                    index_t pages_per_chunk)
  {
    const auto start_page_index = pages_per_chunk - available_pages;

    unsigned int virtual_pos = atomicAdd(&back_, available_pages);
    back_ptr_->enqueueChunk(memory_manager, virtual_pos, chunk_index, available_pages, &back_ptr_, &front_ptr_, &old_ptr_, &old_count_, start_page_index);

    semaphore.signal(available_pages);
    return true;
  }

  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename Desc,typename MemoryManagerType>
  inline void *
  PageQueueVL<CHUNK_TYPE>::allocPage(const Desc& d,MemoryManagerType *memory_manager)
  {
    using ChunkType = typename MemoryManagerType::ChunkType;

    MemoryIndex index;
    uint32_t chunk_index;
    auto pages_per_chunk = MemoryManagerType::QI::getPagesPerChunkFromQueueIndex(queue_index_);

    semaphore.wait(d,1, pages_per_chunk, [&]()
    {
      if (!memory_manager->template allocateChunk<false>(chunk_index))
        {
          if(!FINAL_RELEASE)
            d.out<<"TODO: Could not allocate chunk!!!\n";
        }

      ChunkType::initializeChunk(memory_manager->d_data, chunk_index, pages_per_chunk);
      sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::device);
      enqueueChunk(d,memory_manager, chunk_index, pages_per_chunk);
    });

    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);
    sycl::group_barrier(d.item.get_sub_group());
    
    unsigned int virtual_pos = Ouro::atomicAggInc(&front_);
    front_ptr_->template dequeue<Desc,QueueChunkType::DEQUEUE_MODE::DEQUEUE>
      (d,memory_manager, virtual_pos, index.index, &front_ptr_, &old_ptr_, &old_count_);
    if (index.index==DeletionMarker<uint32_t>::val || index.index==0)
      return nullptr; // failed to allocated page in time
    
    chunk_index = index.getChunkIndex();
    return ChunkType::getPage(memory_manager->d_data, chunk_index, index.getPageIndex(), page_size_);
  }

  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename Desc,typename MemoryManagerType>
  inline void
  PageQueueVL<CHUNK_TYPE>::freePage(const Desc& d,MemoryManagerType *memory_manager,
                                    MemoryIndex index)
  {
    enqueue(d,memory_manager, index.index);

    sycl::atomic_fence(sycl::memory_order::seq_cst,
                       sycl::memory_scope::work_group);

    semaphore.signal(1);
  }

  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename Desc,typename MemoryManagerType>
  inline void
  PageQueueVL<CHUNK_TYPE>::enqueue(const Desc& d,MemoryManagerType *memory_manager,
                                   index_t index)
  {
    // Increase back and compute the position on a chunk
    // const unsigned int virtual_pos = atomicAdd(&back_, 1);
    unsigned int virtual_pos = Ouro::atomicAggInc(&back_);
    back_ptr_->enqueue(d, memory_manager, virtual_pos, index, &back_ptr_, &front_ptr_, &old_ptr_, &old_count_);
  }
}
