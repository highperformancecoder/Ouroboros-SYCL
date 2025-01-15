#include <sycl/sycl.hpp>
#pragma once
#include "Parameters.h"
#include "device/queues/page/PageQueue.dp.hpp"
#include "device/MemoryIndex.dp.hpp"
#include "device/BulkSemaphore_impl.dp.hpp"

namespace Ouro
{
  // ##############################################################################################################################################
  //
  template <typename ChunkType>
  template <typename Desc,typename MemoryManagerType>
  inline void
  PageQueue<ChunkType>::init(const Desc& d,MemoryManagerType *memory_manager)
  {
    //for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size_; i += blockDim.x * gridDim.x)
    auto offs=d.item.get_global_linear_id();
    auto range=d.item.get_global_range(0);
    for (int i = offs; i < size_; i += range)
      {
        queue_[i] = DeletionMarker<index_t>::val;
      }
  }

  // ##############################################################################################################################################
  //
  template <typename ChunkType>
  template <typename Desc>
  inline bool PageQueue<ChunkType>::enqueue(const Desc& d,index_t chunk_index)
  {
    if (semaphore.signal(1) < size_)
      {
        //we have to wait in case there is still something in the spot
        // note: as the filllevel could be increased by this thread, we are certain that the spot will become available
        // unsigned int pos = Ouro::modPower2<size_>(atomicAdd(&back_, 1));
        unsigned int pos = Ouro::modPower2<size_>(Ouro::atomicAggInc(&back_));
        //                d.out<<"PageQueue<ChunkType>::enqueue "<<chunk_index<<" pos "<<pos<<" size_ "<<size_<<sycl::endl;
        while (atomicCAS(queue_ + pos, DeletionMarker<index_t>::val, chunk_index) != DeletionMarker<index_t>::val)
          Ouro::sleep();
        return true;
      }
    return false;
  }

  // ##############################################################################################################################################
  //
  template <typename ChunkType>
  template <typename MemoryManagerType>
  inline bool PageQueue<ChunkType>::enqueueInitialChunk(
                                                                 MemoryManagerType *memory_manager, index_t chunk_index, int availablePages,
                                                                 index_t pages_per_chunk)
  {
    const auto start_page_index = pages_per_chunk - availablePages;

    for(auto i = start_page_index; i < pages_per_chunk; ++i)
      {
        queue_[back_++] = MemoryIndex::createIndex(chunk_index, i);
      }

    semaphore.signal(availablePages);
    return true;
  }

  // ##############################################################################################################################################
  //
  template <typename ChunkType>
  template <typename Desc>
  inline bool PageQueue<ChunkType>::enqueueChunk(const Desc& d,index_t chunk_index,
                                                          index_t pages_per_chunk)
  {
    if (semaphore.signalExpected(pages_per_chunk) < size_)
      {
        //we have to wait in case there is still something in the spot
        // note: as the filllevel could be increased by this thread, we are certain that the spot will become available
        unsigned int pos = Ouro::modPower2<size_>(atomicAdd(&back_, pages_per_chunk));
        for(auto i = 0; i < pages_per_chunk; ++i)
          {
            index_t index = MemoryIndex::createIndex(chunk_index, i);
            while (atomicCAS(queue_ + pos, DeletionMarker<index_t>::val, index) != DeletionMarker<index_t>::val)
              {
                Ouro::sleep();
              }
				
            pos = Ouro::modPower2<size_>(++pos);
          }
        return true;
      }
    //no space to enqueue -> fail
    return false;
  }

  // ##############################################################################################################################################
  //
  template <typename ChunkType>
  template <typename Desc>
  inline void PageQueue<ChunkType>::dequeue(const Desc& d, MemoryIndex &index)
  {
    // Dequeue from queue
    // #if (__CUDA_ARCH__ < 700)
    // unsigned int pos = Ouro::modPower2<size_>(atomicAdd(&front_, 1));
    // #else
    unsigned int pos = Ouro::modPower2<size_>(Ouro::atomicAggInc(&front_));
    // #endif
    auto counter {0U};
    while ((index.index = atomicExch(queue_ + pos, DeletionMarker<index_t>::val)) == DeletionMarker<index_t>::val)
      {
        Ouro::sleep(counter++);
        if(counter++ > 1000000)
          {
            int count, reserved,expected;
            semaphore.getValues(count, reserved, expected);
            d.out<<d.item.get_group_linear_id()<<" - "<<d.item.get_local_linear_id()<<" - "<<d.item.get_sub_group().get_local_linear_id()<<
              " We died in PageQueue::dequeue waiting on a value! :( pos: "<<pos<<" | "<<count<<" - "<<
              reserved<<" - "<<expected<<sycl::endl;
            return;
          }
      }
  }

  // ##############################################################################################################################################
  //

  template <typename ChunkType>
  template <typename Desc,typename MemoryManagerType>
  inline void *
  PageQueue<ChunkType>::allocPage(const Desc& d,MemoryManagerType *memory_manager)
  {
    MemoryIndex index;
    uint32_t chunk_index;
    auto pages_per_chunk = MemoryManagerType::QI::getPagesPerChunkFromQueueIndex(queue_index_);
    semaphore.wait(d, 1, pages_per_chunk, [&]()
    {
      if (!memory_manager->template allocateChunk<false>(chunk_index))
        {
          if(!FINAL_RELEASE)
            d.out<<"TODO: Could not allocate chunk!!!\n";
        }

      ChunkType::initializeChunk(memory_manager->d_data, chunk_index, pages_per_chunk);
      enqueueChunk(d,chunk_index, pages_per_chunk);
    });
	
    // Get index from queue
    dequeue(d,index);

    // Return page to caller
    return ChunkType::getPage(memory_manager->d_data, index.getChunkIndex(), index.getPageIndex(), page_size_);
  }

  // ##############################################################################################################################################
  //
  template <typename ChunkType>
  template <typename Desc,typename MemoryManagerType>
  inline void
  PageQueue<ChunkType>::freePage(const Desc& d,MemoryManagerType *memory_manager,
                                 MemoryIndex index)
  {
    // Enqueue this index into the queue
    enqueue(d,index.index);
  }
}
