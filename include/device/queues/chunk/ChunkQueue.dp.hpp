#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once

#include "Definitions.h"
#include "device/ChunkAccess.dp.hpp"
#include "device/BulkSemaphore.dp.hpp"
#include "Parameters.h"

namespace Ouro
{
  // Forward declaration
  struct MemoryIndex;

  template <typename CHUNK_TYPE>
  struct ChunkQueue
  {
    using SemaphoreType = BulkSemaphore;
    using ChunkType = CHUNK_TYPE;

    // Members
    index_t* queue_;
    SemaphoreType semaphore{ SemaphoreType::null_value };
    int count_{ 0 };
    unsigned int front_{ 0 };
    unsigned int back_{ 0 };
    int queue_index_{ 0 };
    int page_size_{ 0 };

    // Static Members
    static constexpr bool virtualized{false};
    static constexpr int size_{ page_queue_size };
    static_assert(Ouro::isPowerOfTwo(size_), "Chunk Queue size is not Power of 2!");
    static constexpr int lower_fill_level{static_cast<int>(static_cast<float>(size_) * LOWER_FILL_LEVEL_PERCENTAGE)};

    // Methods
    template <class Desc>
    __dpct_inline__ bool enqueue(const Desc&,index_t chunk_index, ChunkType *chunk);

    template <class Desc>
    __dpct_inline__ bool enqueueChunk(const Desc&,index_t chunk_index,
                                      index_t pages_per_chunk,
                                      ChunkType *chunk);

    template <typename Desc,typename MemoryManagerType>
    __dpct_inline__ void init(const Desc&,MemoryManagerType *memory_manager);

    template <typename MemoryManagerType>
    __dpct_inline__ bool
    enqueueInitialChunk(MemoryManagerType *memory_manager,
                        index_t chunk_index, int available_pages,
                        index_t pages_per_chunk);

    template <typename Desc,typename MemoryManagerType>
    __dpct_inline__ void *allocPage(const Desc&,MemoryManagerType *memory_manager);

    template <typename Desc,typename MemoryManagerType>
    __dpct_inline__ void freePage(const Desc&,MemoryManagerType *memory_manager,
                                  MemoryIndex index);

    template <typename MemoryManagerType>
    __dpct_inline__ bool
    preFillQueue(MemoryManagerType *memory_manager, index_t chunk_index,
                 index_t pages_per_chunk, ChunkType *chunk = nullptr)
    {
      return enqueueChunk(chunk_index, pages_per_chunk, chunk);
    }

    void resetQueue()
    {
      count_ = 0;
      semaphore = SemaphoreType::null_value;
      front_ = 0;
      back_ = 0;
    }

    __dpct_inline__ uint32_t getCount()
    {
      return semaphore.getCount();
    }
  };
}
