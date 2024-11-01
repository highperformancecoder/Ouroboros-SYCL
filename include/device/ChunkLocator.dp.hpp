#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once

#include "Utility.h"

struct ChunkLocator
{
	/*!	\brief						Initialize all indices to 0
	 *	\return						Void
	 *	\param[in]	num_chunks		Number of chunks */
        __dpct_inline__ void init(unsigned int num_chunks,
                                  const sycl::nd_item<1> &item_ct1)
        {
                for (int i =
                         item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                         item_ct1.get_local_id(0);
                     i < Ouro::divup(num_chunks, num_bits);
                     i +=
                     item_ct1.get_local_range(0) * item_ct1.get_group_range(0))
                {
			d_chunk_flags[i] = 0;
		}
	}

	/*!	\brief						Set corresponding bit for chunk index
	 *	\return						Void
	 *	\param[in]	chunk_index		Which chunk to set */
        __dpct_inline__ void initChunkIndex(unsigned int chunk_index)
        {
          //atomicOr(&d_chunk_flags[chunk_index >> division_factor], 1 << Ouro::modPower2<num_bits>(chunk_index));
          Ouro::Atomic<int>(d_chunk_flags[chunk_index >> division_factor]) |= 1 << Ouro::modPower2<num_bits>(chunk_index);
	}

	/*!	\brief						Given a potential chunk_index, check if it is a correct index, otherwise search lower until we find one
	 *	\return						return valid index
	 *	\param[in]	chunk_index		Potential chunk index that needs to be checked */
        __dpct_inline__ unsigned int getChunkIndex(unsigned int chunk_index)
        {
		auto index = chunk_index >> division_factor; // Get index position
		// printf("%d - %d | Chunk_Index: %u - Index: %u\n", threadIdx.x, blockIdx.x, chunk_index, index);
		auto mask = (1U << (Ouro::modPower2<num_bits>(chunk_index) + 1)) - 1; // Only look at the bits from the index position down so we can use built-ins
		while(true)
		{
                  auto local_index = num_bits - sycl::clz(d_chunk_flags[index] & mask); // Find the first bit set from the top down
			// printf("%d - %d | Local Index: %u\n", threadIdx.x, blockIdx.x, local_index - 1);
			if(local_index)
				return (index << division_factor) + (local_index - 1); // Index is 1-based (0 would mean nothing found)

			// Go back, set mask to full
			if(index == 0)
			{
//				if(!FINAL_RELEASE)
//					printf("Oh no! %u %u\n", mask, chunk_index);
                                assert(0);
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
