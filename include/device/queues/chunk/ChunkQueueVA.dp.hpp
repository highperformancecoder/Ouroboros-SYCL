#include <sycl/sycl.hpp>
#pragma once

#include "Parameters.h"
#include "Definitions.h"
#include "device/ChunkAccess.dp.hpp"
#include "device/BulkSemaphore.dp.hpp"
#include "device/queues/QueueChunk.dp.hpp"

namespace Ouro
{
  // Forward declaration
  struct MemoryIndex;

  template <typename CHUNK_TYPE>
  struct ChunkQueueVA
  {
    using SemaphoreType = BulkSemaphore;
    using ChunkType = CHUNK_TYPE;
    using Base = typename ChunkType::Base;
    using QueueChunkType = QueueChunk<Base>;

    // Members
    index_t* queue_{nullptr}; // Chunk-Pointers
    int queue_index_{ 0 }; // Which queue is this!
    int page_size_{ 0 };
    int count_{ 0 }; // How many chunk pointers do we have?
    unsigned int front_{ 0 }; // Current front (virtual)
    unsigned int back_{ 0 }; // Current back (virtual)
    SemaphoreType semaphore{ SemaphoreType::null_value }; // Access Management

    // Static Members
    static constexpr bool virtualized{true};
    static constexpr int size_{ virtual_queue_size }; // How many chunk pointers can we store?
    static constexpr int num_spots_{ virtual_queue_size * QueueChunkType::num_spots_ }; // How many virtual spots do we have? Reduce by one so enqueue can't interfere with dequeue
    static constexpr int lower_fill_level{static_cast<int>(static_cast<float>(num_spots_) * LOWER_FILL_LEVEL_PERCENTAGE)};
    static_assert(Ouro::isPowerOfTwo(num_spots_), "Virtualized Queue size is not Power of 2!");

    // Methods
    template <typename MemoryManagerType>
    inline bool preFillQueue(MemoryManagerType *memory_manager,
                                      index_t chunk_index,
                                      index_t pages_per_chunk)
    {
      return enqueueChunk(memory_manager, chunk_index, pages_per_chunk);
    }

    inline uint32_t getCount() const
    {
      return semaphore.getCount();
    }

    inline index_t computeChunkID(index_t virtual_position)
    {
      return (virtual_position / QueueChunkType::num_spots_) % size_;
    }

    template <typename Desc,typename MemoryManagerType>
    inline void init(const Desc&,MemoryManagerType *memory_manager);

    template <typename Desc,typename MemoryManagerType>
    inline void *allocPage(const Desc&,MemoryManagerType *memory_manager);

    template <typename Desc,typename MemoryManagerType>
    inline void freePage(const Desc&,MemoryManagerType *memory_manager,
                                  MemoryIndex index);

    template <typename Desc,typename MemoryManagerType>
    inline QueueChunkType *
    accessQueueElement(const Desc&,MemoryManagerType *memory_manager, index_t chunk_id,
                       index_t v_position);
    template <typename Desc,typename MemoryManagerType>
    inline void
    enqueue(const Desc&,MemoryManagerType *memory_manager, index_t index,
            typename MemoryManagerType::ChunkType *chunk);
    template <typename Desc,typename MemoryManagerType>
    inline bool
    enqueueChunk(const Desc&,MemoryManagerType *memory_manager, index_t chunk_index,
                 index_t pages_per_chunk,
                 typename MemoryManagerType::ChunkType *chunk);
  };
}
