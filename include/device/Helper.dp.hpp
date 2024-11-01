#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#pragma once

struct AllocationHelper
{
        template <typename T> static __dpct_inline__ T getNextPow2(T n)
        {
		return 1 << (getNextPow2Pow(n));
	}

        template <typename T> static __dpct_inline__ int getNextPow2Pow(T n)
        {
		if ((n & (n - 1)) == 0)
                  return 32 - sycl::clz(n) - 1;
		else
                  return 32 - sycl::clz(n);
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

        static __dpct_inline__ int getQueueIndex(size_t size)
        {
                return std::max(AllocationHelper::getNextPow2Pow(size) -
                                    smallest_page_factor,
                                0);
        }

        static __dpct_inline__ int getPageSize(size_t size)
        {
          return std::max(AllocationHelper::getNextPow2(size), static_cast<size_t>(SMALLEST_PAGE_SIZE));
	}

        static __dpct_inline__ size_t getPagesPerChunk(size_t size)
        {
		return SIZE / getPageSize(size);
	}

        static __dpct_inline__ size_t
        getPagesPerChunkFromQueueIndex(unsigned int index)
        {
		return pages_per_chunk_factor >> index;
	}

        static __dpct_inline__ int getPageSizeFromQueueIndex(unsigned int index)
        {
		return SMALLEST_PAGE_SIZE << index;
	}
};
