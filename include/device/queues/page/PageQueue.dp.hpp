#include <sycl/sycl.hpp>
#pragma once

#include "Definitions.h"
#include "device/BulkSemaphore.dp.hpp"
#include "Parameters.h"
#include "device/PageIndexChunk.dp.hpp"

namespace Ouro
{
  // Forward declaration
  struct MemoryIndex;

  template <typename CHUNK_TYPE>
  struct PageQueue
  {
    using SemaphoreType = BulkSemaphore;
    using ChunkType = CHUNK_TYPE;

    // Members
    index_t* queue_;
    SemaphoreType semaphore{ SemaphoreType::null_value };
    unsigned int front_{ 0 };
    unsigned int back_{ 0 };
    int queue_index_{ 0 };
    int page_size_{ 0 };

    // Static Members
    static constexpr bool virtualized{false};
    static constexpr int size_{ page_queue_size };
    static_assert(Ouro::isPowerOfTwo(size_), "Page Queue size is not Power of 2!");
    static constexpr int lower_fill_level{static_cast<int>(static_cast<float>(size_) * LOWER_FILL_LEVEL_PERCENTAGE)};

    // Methods
    template <class Desc>
    inline bool enqueue(const Desc&,index_t chunk_index);

    template <class Desc>
    inline bool enqueueChunk(const Desc&,index_t chunk_index,
                                      index_t pages_per_chunk);

    template <class Desc>
    inline void dequeue(const Desc&,MemoryIndex &index);

    template <typename Desc,typename MemoryManagerType>
    inline void init(const Desc&,MemoryManagerType *memory_manager);

    template <typename MemoryManagerType>
    inline bool preFillQueue(MemoryManagerType *memory_manager,
                                      index_t chunk_index,
                                      index_t pages_per_chunk)
    {
      return enqueueChunk(chunk_index, pages_per_chunk);
    }

    template <typename MemoryManagerType>
    inline bool
    enqueueInitialChunk(MemoryManagerType *memory_manager,
                        index_t chunk_index, int available_pages,
                        index_t pages_per_chunk);

    template <typename Desc,typename MemoryManagerType>
    inline void *allocPage(const Desc&,MemoryManagerType *memory_manager);

    template <typename Desc,typename MemoryManagerType>
    inline void freePage(const Desc&,MemoryManagerType *memory_manager,
                                  MemoryIndex index);

    void resetQueue()
    {
      semaphore = SemaphoreType::null_value;
      front_ = 0;
      back_ = 0;
    }

    inline uint32_t getCount()
    {
      return semaphore.getCount();
    }
  };
}
