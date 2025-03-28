#include <sycl/sycl.hpp>
#pragma once

#include "Utility.dp.hpp"

namespace Ouro
{
  struct BulkSemaphore
  {
    // | --- Counter (20 + 1 bits) --- | --- Expected (21 bits) --- | --- Reserved (21 bits) --- |

  public:
	
    // ################################################################################################################
    // Primitives
    inline unsigned long long
    create64BitSubAdder_expected(unsigned long long N)
    {
      return ~(N << middle_mask_shift) + 1;
    }

    inline unsigned long long
    create64BitSubAdder_reserved(unsigned long long N)
    {
      return ~(N << upper_mask_shift) + 1;
    }

    inline void getValues(int &count, int &expected, int &reserved) const
    {
      count = getCount();
      expected = static_cast<int>((value >> middle_mask_shift) & (highest_value_mask));
      reserved = static_cast<int>((value >> upper_mask_shift) & (highest_value_mask));
    }

    // Create a new value
    static inline unsigned long long
    createValueExternal(int count, int expected, int reserved)
    {
      return static_cast<unsigned long long>(count) + null_value
        + (static_cast<unsigned long long>(expected) << middle_mask_shift)
        + (static_cast<unsigned long long>(reserved) << upper_mask_shift);
    }

    // Create a new value
    inline void createValueInternal(int count, int expected, int reserved)
    {
      value = createValueExternal(count, expected, reserved);
    }

    inline void read(BulkSemaphore &semaphore)
    {
      semaphore.value = value;
    }

    // ################################################################################################################
    // Static Variables
    static constexpr unsigned long long middle_mask_shift{ 25 };
    static constexpr unsigned long long upper_mask_shift{ middle_mask_shift * 2 };
    // Highest expressable value
    static constexpr unsigned long long highest_value_mask{ (1ULL << middle_mask_shift) - 1 };
    // Additive Subtraction value for 1
    static constexpr unsigned long long subtract_one_value{~(1ULL) + 1};

  public:

    BulkSemaphore() : value{null_value} {}
    BulkSemaphore(unsigned long long init_value) : value{init_value}{}
	
    // Value extractor
    inline int getCount() const
    {
      return static_cast<int>(value & highest_value_mask) - null_value;
    }

    // Try to allocate some resource
    template <typename Desc,typename T>
    inline void wait(const Desc&,int N, uint32_t number_pages_on_chunk,
                              T allocationFunction);
  
    // Try to increase resources
    inline bool tryReduce(int N);

    // Free a resource (number of pages only set for the thread which allocated stuff)
    inline int signalExpected(unsigned long long N);

    // Free a resource (number of pages only set for the thread which allocated stuff)
    inline int signal(unsigned long long N);

    unsigned long long value{null_value};
    static constexpr unsigned long long null_value {(1ULL << (middle_mask_shift - 1))};
  };
}
