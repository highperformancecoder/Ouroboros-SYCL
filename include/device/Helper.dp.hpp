#include <sycl/sycl.hpp>
#include <cmath>
#pragma once

struct AllocationHelper
{
        template <typename T> static inline T getNextPow2(T n)
        {
		return 1 << (getNextPow2Pow(n));
	}

        template <typename T> static inline int getNextPow2Pow(T n)
        {
		if ((n & (n - 1)) == 0)
                  return 8*sizeof(T) - sycl::clz(n) - 1;
		else
                  return 8*sizeof(T) - sycl::clz(n);
	}

	template <typename T, T n>
	static constexpr int static_getNextPow2Pow()
	{
		if ((n & (n - 1)) == 0)
			return 32 - Ouro::static_clz<n>::value - 1;
		else
			return 32 - Ouro::static_clz<n>::value;
	}

	template <typename T, T n>
	static constexpr size_t static_getNextPow2()
	{
		return 1 << (static_getNextPow2Pow(n));
	}
};

template <unsigned int SMALLEST_PAGE_SIZE, unsigned int SIZE>
struct QueueIndex
{
	static constexpr int pages_per_chunk_factor{SIZE / SMALLEST_PAGE_SIZE};
	static constexpr int smallest_page_factor{AllocationHelper::static_getNextPow2Pow<unsigned int, SMALLEST_PAGE_SIZE>()};

        static inline int getQueueIndex(size_t size)
        {
                return std::max(AllocationHelper::getNextPow2Pow(size) -
                                    smallest_page_factor,
                                0);
        }

        static inline int getPageSize(size_t size)
        {
          return std::max(AllocationHelper::getNextPow2(size), static_cast<size_t>(SMALLEST_PAGE_SIZE));
	}

        static inline size_t getPagesPerChunk(size_t size)
        {
		return SIZE / getPageSize(size);
	}

        static inline size_t
        getPagesPerChunkFromQueueIndex(unsigned int index)
        {
		return pages_per_chunk_factor >> index;
	}

        static inline int getPageSizeFromQueueIndex(unsigned int index)
        {
		return SMALLEST_PAGE_SIZE << index;
	}
};
