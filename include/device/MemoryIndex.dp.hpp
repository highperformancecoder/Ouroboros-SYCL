#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once

#include "Parameters.h"
#include "Definitions.h"

namespace Ouro
{
  struct MemoryIndex
  {
    static constexpr unsigned int NumBitsForPage{NUM_BITS_FOR_PAGE};
    static constexpr unsigned int NumBitsForChunk{ 32 - NumBitsForPage };
    static constexpr unsigned int MaxNumChunks{ 1 << NumBitsForChunk };
    static constexpr unsigned int PageBitMask{(1 << NUM_BITS_FOR_PAGE) - 1};
    static constexpr uint32_t MAX_VALUE{0xFFFFFFFF};

    // Data	
    uint32_t index;

    MemoryIndex() : index{0U}{}
    MemoryIndex(uint32_t chunk_index, uint32_t page_index) : index{(chunk_index << NumBitsForPage) + page_index}{}

    // Methods
    // ----------------------------------------------------------------------------
    __dpct_inline__ uint32_t getIndex() { return index; }
    // ----------------------------------------------------------------------------
    __dpct_inline__ void getIndex(uint32_t &chunk_index, uint32_t &page_index)
    {
      const auto temp_index = index;
      chunk_index = temp_index >> NumBitsForPage;
      page_index = temp_index & PageBitMask;
    }
    // ----------------------------------------------------------------------------
    __dpct_inline__ uint32_t getChunkIndex()
    {
      return index >> NumBitsForPage;
    }
    // ----------------------------------------------------------------------------
    __dpct_inline__ uint32_t getPageIndex()
    {
      return index & PageBitMask;
    }
    // ----------------------------------------------------------------------------
    __dpct_inline__ static constexpr uint32_t
    createIndex(uint32_t chunk_index, uint32_t page_index)
    {
      return (chunk_index << NumBitsForPage) + page_index;
    }
    // ----------------------------------------------------------------------------
    __dpct_inline__ void setIndex(uint32_t ind) { index = ind; }
    // ----------------------------------------------------------------------------
    __dpct_inline__ void setIndex(uint32_t chunk_index, uint32_t page_index)
    {
      index = createIndex(chunk_index, page_index);
    }
  };
}
