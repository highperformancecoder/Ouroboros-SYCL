#include <sycl/sycl.hpp>
#pragma once

#include "Utility.h"

namespace Ouro
{
  struct ChunkLocator
  {
    /*!	\brief						Initialize all indices to 0
     *	\return						Void
     *	\param[in]	num_chunks		Number of chunks */
    template <class Desc>
    inline void init(const Desc& d, unsigned int num_chunks)
    {
      for (int i = d.item.get_global_linear_id();
           i < Ouro::divup(num_chunks, num_bits);
           i += d.item.get_global_range().size())
        {
          d_chunk_flags[i] = 0;
        }
    }

    /*!	\brief						Set corresponding bit for chunk index
     *	\return						Void
     *	\param[in]	chunk_index		Which chunk to set */
    inline void initChunkIndex(unsigned int chunk_index)
    {
      //atomicOr(&d_chunk_flags[chunk_index >> division_factor], 1 << Ouro::modPower2<num_bits>(chunk_index));
      Ouro::Atomic<int>(d_chunk_flags[chunk_index >> division_factor]) |= 1 << Ouro::modPower2<num_bits>(chunk_index);
    }

    /*!	\brief						Given a potential chunk_index, check if it is a correct index, otherwise search lower until we find one
     *	\return						return valid index
     *	\param[in]	chunk_index		Potential chunk index that needs to be checked */
    inline unsigned int getChunkIndex(unsigned int chunk_index)
    {
      auto index = chunk_index >> division_factor; // Get index position
      auto mask = (1U << (Ouro::modPower2<num_bits>(chunk_index) + 1)) - 1; // Only look at the bits from the index position down so we can use built-ins
      while(true)
        {
          auto local_index = num_bits - sycl::clz(d_chunk_flags[index] & mask); // Find the first bit set from the top down
          if(local_index)
            return (index << division_factor) + (local_index - 1); // Index is 1-based (0 would mean nothing found)

          // Go back, set mask to full
          if(index == 0)
            {
              //if(!FINAL_RELEASE)
              //  d.out<<"Oh no!"<<mask<<chunk_index<<sycl::endl;
              assert(false && "getChunkIndex");
            }
          --index;
          mask = 0xFFFFFFFF;
        }
    }

    /*!	\brief						How large is the full array of flags
     *	\return						size of array
     *	\param[in]	num_chunks		Number of chunks */
    static constexpr size_t size(unsigned int num_chunks)
    {
      return Ouro::divup(num_chunks, num_bits) * sizeof(int);
    }

    // Member
    int* d_chunk_flags{nullptr};

    // Static Members
    static constexpr int num_bits{Ouro::sizeofInBits<std::remove_pointer<decltype(d_chunk_flags)>::type>()};
    static constexpr int division_factor{countBitShift(num_bits)}; // Divide/Multiply by 32 using shift operator
  };
}
