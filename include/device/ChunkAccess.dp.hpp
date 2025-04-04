#include <sycl/sycl.hpp>
#pragma once

#include "Parameters.h"
#include "Definitions.h"
#include "Utility.dp.hpp"

namespace Ouro
{
  template <size_t SIZE, size_t SMALLEST_PAGE>
  struct ChunkAccess
  {
    using MaskDataType = unsigned int;
    static constexpr size_t Chunk_Size_{SIZE};
    static constexpr size_t smallest_page_size_{SMALLEST_PAGE};
    static constexpr unsigned int MaximumBitMaskSize_{divup<unsigned int>(Chunk_Size_, smallest_page_size_ * sizeof(MaskDataType) * BYTE_SIZE)};

    enum class Mode
      {
        ERROR, SUCCESSFULL, DEQUEUE_CHUNK, RE_ENQUEUE_CHUNK, CONTINUE
      };

    enum class FreeMode
      {
        SUCCESS, FIRST_FREE, DEQUEUE
      };

    // Members
    int count; // Number of available pages
    int size; // Number of pages
    /*
      DPCT1102:0: Zero-length arrays are not permitted in SYCL device code.
    */
    MaskDataType availability_mask[MaximumBitMaskSize_]; // Bitmask of how
    // many are free

    // Methods
    inline ChunkAccess(int available_pages, uint32_t number_pages)
      : count{available_pages}, size{int(number_pages)}
    {
      // Setup Availability mask
      auto taken_pages = number_pages - available_pages;
#pragma unroll
      for (auto i = 0; i < MaximumBitMaskSize_; ++i)
        {
          if(taken_pages >= Ouro::sizeofInBits<MaskDataType>()) 
            {
              availability_mask[i] = 0U;
              taken_pages -= Ouro::sizeofInBits<MaskDataType>();
            }
          else
            {
              // Set the rest of the bits (set the next greater bit for taken pages, -1 gives bitmask with 1 for taken pages, invert that
              if(number_pages >= Ouro::sizeofInBits<MaskDataType>())
                {
                  availability_mask[i] = ~((1U << (taken_pages)) - 1);
                }
              else
                {
                  // Mask out top bits
                  availability_mask[i] = ~((1U << (taken_pages)) - 1) & ((1U << number_pages) - 1);
                }
              taken_pages = 0;
            }

          if(number_pages > Ouro::sizeofInBits<MaskDataType>())
            {
              number_pages -= Ouro::sizeofInBits<MaskDataType>();
            }
          else
            {
              number_pages = 0;
            }
        }
    }

    inline ChunkAccess(uint32_t number_pages)
      : count{static_cast<int>(number_pages)}, size{int(number_pages)}
    {
      // Setup Availability mask
#pragma unroll
      for (auto i = 0; i < MaximumBitMaskSize_; ++i)
        {
          // Set the rest of the bits (set the next greater bit for taken pages, -1 gives bitmask with 1 for taken pages, invert that
          if(number_pages >= Ouro::sizeofInBits<MaskDataType>())
            {
              // All bits set
              availability_mask[i] = ~0U;
            }
          else
            {
              // Mask out top bits
              availability_mask[i] = (1U << number_pages) - 1;
            }

          number_pages = (number_pages > Ouro::sizeofInBits<MaskDataType>()) ? (number_pages - Ouro::sizeofInBits<MaskDataType>()) : 0;
        }
    }

    inline MaskDataType createBitPattern(int bit)
    {
      return ~(1U << bit);
    }

    inline bool checkBitSet(MaskDataType value, int bit)
    {
      return (1U << bit) & value;
    }

    // Free up a page, manipulating the count and availability mask, locking required
    template <class Desc>
    inline FreeMode freePage(const Desc&,index_t queue_index);

    // Try to allocate a page, locking required
    template <class Desc>
    inline Mode allocPage(const Desc&,index_t &page_index);

    inline bool tryFlashChunk();
  };
}
