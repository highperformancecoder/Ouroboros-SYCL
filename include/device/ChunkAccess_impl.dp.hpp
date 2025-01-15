#include <sycl/sycl.hpp>
#pragma once

#include "device/ChunkAccess.dp.hpp"

namespace Ouro
{
  // ##############################################################################################################################################
  //
  template <size_t SIZE, size_t SMALLEST_PAGE>
  template <typename Desc>
  inline typename ChunkAccess<SIZE, SMALLEST_PAGE>::FreeMode
  ChunkAccess<SIZE, SMALLEST_PAGE>::freePage(const Desc&,index_t page_index)
  {
    const int mask_index = page_index / (Ouro::sizeofInBits<MaskDataType>());
    const int local_page_index = page_index % (Ouro::sizeofInBits<MaskDataType>());
    const auto bit_pattern = 1U << local_page_index;
    // Set bit to 1
    atomicOr(&availability_mask[mask_index], bit_pattern);
	
    // Please do NOT reorder here
    //__threadfence_block();
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::work_group);

    auto current_count = atomicAdd(&count, 1) + 1;
    if (current_count == 1)
      return FreeMode::FIRST_FREE;
    else if(current_count == size)
      return FreeMode::DEQUEUE;
    return FreeMode::SUCCESS;
  }

  // ##############################################################################################################################################
  //
  template <size_t SIZE, size_t SMALLEST_PAGE>
  inline bool ChunkAccess<SIZE, SMALLEST_PAGE>::tryFlashChunk()
  {
    // Try to reduce count to 0, if previous value is != size, someone tries do allocate from this chunk right now!
    return atomicCAS(&count, size, 0) == size;
  }

  // ##############################################################################################################################################
  //
  template <size_t SIZE, size_t SMALLEST_PAGE>
  template <typename Desc>
  inline typename ChunkAccess<SIZE, SMALLEST_PAGE>::Mode
  ChunkAccess<SIZE, SMALLEST_PAGE>::allocPage(const Desc& d,index_t &page_index)
  {
    int current_count{ 0 };
    auto mode = Mode::SUCCESSFULL;
    while ((current_count = atomicSub(&count, 1)) <= 0)
      {
        if((current_count = atomicAdd(&count, 1)) < 0)
          return Mode::CONTINUE;

        // If we observed 0 -> 1, we potentially want to re-enqueue this chunk
        if(current_count == 0)
          mode = Mode::RE_ENQUEUE_CHUNK;
      }
    if(current_count == 1)
      {
        // We take the last page (so just take the page and don't re-enqueue) 
        // OR 
        // We just want to dequeue this chunk
        // TODO: Not sure if this logic is completely right, if not both modi need the DEQUEUE_CHUNK
        // mode = (mode == Mode::RE_ENQUEUE_CHUNK) ? Mode::SUCCESSFULL : Mode::DEQUEUE_CHUNK;
        mode = Mode::DEQUEUE_CHUNK;
      }
    // else
    // {
    // 	// If we had a count larger than 1, we can either simply do as normal or stay in the re-enqueue mode
    // 	if(mode != Mode::RE_ENQUEUE_CHUNK)
    // 		mode = Mode::SUCCESSFULL;
    // }

    int least_significant_bit{ 0 };

    // Offset in the range of 0-63
    //const int offset = (threadIdx.x + blockIdx.x) % Ouro::sizeofInBits<MaskDataType>();
    const int offset = (d.item.get_local_linear_id() + d.item.get_group_linear_id()) % Ouro::sizeofInBits<MaskDataType>();

    // TODO: Why is this not faster instead of always using the full mask?
    // int mask = Ouro::divup(size, sizeof(MaskDataType) * BYTE_SIZE);
    //int bitmask_index = threadIdx.x;
    int bitmask_index = d.item.get_local_linear_id();

    // There is a reason why this is a while true loop and not just a loop over all MAXIMUM_BITMASK_SIZE entries
    // Imagine we have 2 threads, currently one page in mask 3
    // One thread decrements the count and starts at the first mask to look for the bit -> does not find any
    // One thread then frees a page on the first mask
    // A second thread decrements the count and starts looking at mask 3 -> finds the bit immediately
    // The first thread would now look at mask 2 - 3 - 4 ... and not find the bit on mask 1, as it already looked there
    // Hence, we need a while(true) loop, since we are guaranteed to find a bit, but not guaranteed that someone steals our bit
    unsigned int iters{0U};

    //d.out<<"MaximumBitMaskSize_="<<MaximumBitMaskSize_<<sycl::endl;
    
    while(true)
    //while(++iters<100000)
      {
        // We want each thread starting at a different position, for this we do a circular shift
        // This way we can still use the build in __ffsll but will still start our search at different 
        // positions
        // Load mask -> shift by offset to the right and then append whatever was shifted out at the top
        auto current_mask = atomicAdd(&availability_mask[(++bitmask_index) % MaximumBitMaskSize_],0);
        auto without_lower_part = current_mask >> offset;
        auto final_mask = without_lower_part | (current_mask << (Ouro::sizeofInBits<MaskDataType>() - offset));
        //while((least_significant_bit = __ffsll(final_mask))
        // original code was above, but least_significant_bit==0 => final_mask=0
        while (final_mask)
          {
            //--least_significant_bit; // Get actual bit position (as bit 0 return 1)
            least_significant_bit=sycl::ctz(final_mask);
            least_significant_bit = ((least_significant_bit + offset) % Ouro::sizeofInBits<MaskDataType>()); // Correct for shift
            page_index = Ouro::sizeofInBits<MaskDataType>() * (bitmask_index % MaximumBitMaskSize_) // which mask
              + least_significant_bit; // which page on mask

            // Please do NOT reorder here
            //__threadfence_block();
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::work_group);

            auto bit_pattern = createBitPattern(least_significant_bit);
            current_mask = atomicAnd(&availability_mask[bitmask_index % MaximumBitMaskSize_], bit_pattern);

            // Please do NOT reorder here
            //__threadfence_block();
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::work_group);

            if(checkBitSet(current_mask, least_significant_bit))
              {
                // Hehe, we were the one who set this bit :-)
                return mode;
              }
            without_lower_part = current_mask >> offset;
            final_mask = without_lower_part | (current_mask << (Ouro::sizeofInBits<MaskDataType>() - offset));
          }
      }

    // ##############################################################################################################
    // Error Checking
    if(!FINAL_RELEASE)
      {
        d.out<<"We should have gotten a page, but there was nothing for threadId "<<
          d.item.get_local_linear_id()<<" and blockId "<<
          d.item.get_group_linear_id()<<
          " - current count : "<<current_count<<
          " - bitmask_index: "<<bitmask_index<<sycl::endl;
      }
    return Mode::ERROR;
  }
}
