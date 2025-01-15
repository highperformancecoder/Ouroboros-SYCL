#include <sycl/sycl.hpp>
//#include <dpct/dpct.hpp>
#pragma once

#include "device/Chunk.dp.hpp"

namespace Ouro
{
  template <typename ChunkBase, size_t SIZE>
  struct PageChunk : public CommonChunk
  {
    using Base = ChunkBase;
    static constexpr size_t size_{SIZE};
    static constexpr size_t meta_data_size_{CHUNK_METADATA_SIZE};

    // Members
    uint32_t number_pages;

    // ##########################################################################################################################
    // ##########################################################################################################################
    // Methods
    // ##########################################################################################################################
    // ##########################################################################################################################
    PageChunk(const unsigned int page_size, uint32_t number_pages) : CommonChunk(page_size), number_pages{number_pages} {}

    inline void *getPage(memory_t *memory, index_t chunk_index,
                                  uint32_t page_index)
    {
      return reinterpret_cast<void*>(reinterpret_cast<memory_t*>(Base::getData(memory, chunk_index)) + (page_index * page_size));
    }

    // ##########################################################################################################################
    // ##########################################################################################################################
    // STATIC Methods
    // ##########################################################################################################################
    // ##########################################################################################################################

    static constexpr inline size_t size() {
      return meta_data_size_ + size_;
    }

    static inline void *getData(memory_t *memory,
                                         const index_t chunk_index)
    {
      return Base::getData(memory, chunk_index);
    }

    static inline void *getPage(memory_t *memory,
                                         const index_t chunk_index,
                                         const uint32_t page_index,
                                         const unsigned int page_size)
    {
      return Base::getPage(memory, chunk_index, page_index, page_size);
    }

    template <typename QI>
    static inline index_t
    getQueueIndexFromPage(memory_t *memory, index_t chunk_index)
    {
      auto chunk = reinterpret_cast<PageChunk*>(Base::getMemoryAccess(memory, chunk_index));
      return QI::getQueueIndex(chunk->page_size);
    }

    // ##############################################################################################################################################
    //
    static inline PageChunk *getAccess(memory_t *memory,
                                                index_t chunk_index)
    {
      return Base::template getMemoryAccess<PageChunk>(memory, chunk_index);
    }

    // ##############################################################################################################################################
    // Initializer
    static inline PageChunk *initializeChunk(memory_t *memory,
                                                      index_t chunk_index,
                                                      uint32_t number_pages)
    {
      static_assert(Ouro::alignment(sizeof(PageChunk)) <= meta_data_size_, "PageChunk is larger than alignment!");
      /*
        DPCT1109:0: The usage of dynamic memory allocation and
        deallocation APIs cannot be called in SYCL device code. You need
        to adjust the code.
      */
      return new (
                  reinterpret_cast<char *>(getAccess(memory, chunk_index)))
        PageChunk((size_ / number_pages), number_pages);
    }

    // ##############################################################################################################################################
    // Initializer
    static inline PageChunk *
    initializeChunk(memory_t *memory, index_t chunk_index,
                    const int available_pages, uint32_t number_pages)
    {
      static_assert(Ouro::alignment(sizeof(PageChunk)) <= meta_data_size_, "PageChunk is larger than alignment!");
      return new(Base::template getMemoryAccess<memory_t>(memory, chunk_index)) PageChunk((size_ / number_pages), number_pages);
    }
  };
}
