#pragma once
#include <sycl/sycl.hpp>
#include "Ouroboros.dp.hpp"
#include "device/queues/Queues_impl.dp.hpp"
#include "device/queues/chunk/ChunkQueue_impl.dp.hpp"

namespace Ouro
{
  // ##############################################################################################################################################
  //
  //
  // CHUNKS
  //
  //
  // ##############################################################################################################################################

  // ##############################################################################################################
  //
  template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE,
            unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
  template <typename Desc>
  inline void *OuroborosChunks<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE,
                                        NUMBER_QUEUES>::allocPage(const Desc& d,size_t size)
  {
    if(statistics_enabled)
      Ouro::Atomic<unsigned>(stats.pageAllocCount)++;

    // Allocate from chunks
    return d_storage_reuse_queue[QI::getQueueIndex(size)].allocPage(d,this);

  }

  // ##############################################################################################################
  //
  template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE,
            unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
  template <typename Desc>
  inline void
  OuroborosChunks<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::freePage
  (const Desc& d, MemoryIndex index)
  {
    if(statistics_enabled)
      atomicAdd(&stats.pageFreeCount, 1);

    // Deallocate page in chunk
    d_storage_reuse_queue[QueueType::ChunkType::template getQueueIndexFromPage<QI>(d_data, index.getChunkIndex())].freePage(d,this, index);
  }

  // ##############################################################################################################################################
  //
  //
  // PAGES
  //
  //
  // ##############################################################################################################################################

  // ##############################################################################################################
  //
  template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE,
            unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
  template <typename Desc>
  inline void *
  OuroborosPages<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::allocPage(const Desc& d, size_t size)
  {
    if(statistics_enabled)
      Ouro::Atomic<unsigned>(stats.pageAllocCount)++;

    // Allocate from pages
    return d_storage_reuse_queue[QI::getQueueIndex(size)].allocPage(d,this);
  }

  // ##############################################################################################################
  //
  template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE,
            unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
  template <typename Desc>
  inline void
  OuroborosPages<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::freePage
  (const Desc& d, MemoryIndex index)
  {
    if(statistics_enabled)
      Ouro::Atomic<unsigned>(stats.pageFreeCount)++;

    // Deallocate page in chunk
    d_storage_reuse_queue[QueueType::ChunkType::template getQueueIndexFromPage<QI>(d_data, index.getChunkIndex())].freePage(d, this, index);
  }

  // ##############################################################################################################################################
  //
  //
  // OUROBOROS
  //
  //
  // ##############################################################################################################################################

  // ##############################################################################################################################################
  //
  template <class OUROBOROS, class... OUROBOROSES>
  template <typename Desc>
  inline void *Ouroboros<OUROBOROS, OUROBOROSES...>::malloc(const Desc& d, size_t size)
  {
    if(size <= ConcreteOuroboros::LargestPageSize_)
      {
        return memory_manager.allocPage(d, size);
      }
    return next_memory_manager.malloc(d, size);
  }

  // ##############################################################################################################################################
  //
  template <class OUROBOROS, class... OUROBOROSES>
  template <typename Desc>
  inline void Ouroboros<OUROBOROS, OUROBOROSES...>::free(const Desc& d, void *ptr)
  {
    if(!validOuroborosPointer(ptr))
      {
        if(!FINAL_RELEASE && printDebug)
          d.out<<"Freeing CUDA Memory!\n";
        return;
      }
    auto chunk_index = ChunkBase::getIndexFromPointer(memory.d_data, ptr);
    auto revised_chunk_index = memory.chunk_locator.getChunkIndex(chunk_index);
    auto chunk = reinterpret_cast<CommonChunk*>(ConcreteOuroboros::ChunkBase::getMemoryAccess(memory.d_data, revised_chunk_index));
    auto page_size = chunk->page_size;
    unsigned int page_index = (reinterpret_cast<unsigned long long>(ptr) - reinterpret_cast<unsigned long long>(chunk) - ChunkBase::meta_data_size_) / page_size;
    return freePageRecursive(d, page_size, MemoryIndex(revised_chunk_index, page_index));
  }

  // ##############################################################################################################################################
  //
  template <class OUROBOROS, class... OUROBOROSES>
  template <typename Desc>
  inline void
  Ouroboros<OUROBOROS, OUROBOROSES...>::freePageRecursive(const Desc& d, unsigned int page_size,
                                                          MemoryIndex index)
  {
    if(page_size <= ConcreteOuroboros::LargestPageSize_)
      {
        return memory_manager.freePage(d,index);
      }
    return next_memory_manager.freePageRecursive(d, page_size, index);
  }

  // ##############################################################################################################################################
  //
  //
  // HOST
  //
  //
  // ##############################################################################################################################################

  // ##############################################################################################################################################
  //
  template <typename MemoryManagerType>
  void updateMemoryManagerHost(sycl::queue& queue, MemoryManagerType& memory_manager)
  {
    queue.memcpy(&memory_manager, memory_manager.memory.d_memory,
                 sizeof(memory_manager))
      .wait();
  }

  // ##############################################################################################################################################
  //
 template <typename MemoryManagerType>
 void updateMemoryManagerDevice(sycl::queue& queue, MemoryManagerType& memory_manager)
 {
   queue.memcpy(memory_manager.memory.d_memory, &memory_manager,
                sizeof(memory_manager))
     .wait();
 }
}
