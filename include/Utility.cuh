#pragma once
#include "Utility.h"

namespace Ouro
{
	__forceinline__ __device__ unsigned int ldg_cg(const unsigned int* src)
	{
                return *src;
	}

	__forceinline__ __device__ int ldg_cg(const int* src)
        {
                return *src;
        }

	__forceinline__ __device__ unsigned long long ldg_cg(const unsigned long long* src)
	{
          return *src;
	}

	__forceinline__ __device__ const unsigned int& stg_cg(unsigned int* dest, const unsigned int& src)
	{
                *dest = src;
                return src;
        }

	__forceinline__ __device__ void store(uint4* dest, const uint4& src)
	{
          *dest = src;
	}

	__forceinline__ __device__ void store(uint2* dest, const uint2& src)
	{
          *dest = src;
	}

	static __forceinline__ __device__ int lane_id()
	{
		return threadIdx.x & (WARP_SIZE - 1);
	}

	__forceinline__ __device__ void sleep(unsigned int factor = 1)
	{
		__threadfence();
	}

	__forceinline__ __device__ int atomicAggInc(unsigned int *ptr)
	{
		return atomicAdd(ptr, 1);
	}
}

