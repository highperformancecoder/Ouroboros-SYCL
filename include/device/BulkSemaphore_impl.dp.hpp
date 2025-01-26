#pragma once

#include <sycl/sycl.hpp>
#include "BulkSemaphore.dp.hpp"

namespace Ouro
{
  // ##############################################################################################################################################
  //
  inline bool BulkSemaphore::tryReduce(int N)
  {
    // Reduce by N-1
    uint32_t atomic_ret_val = atomicAdd(&value, Ouro::create2Complement(N)) & highest_value_mask;
    if((atomic_ret_val - N) < null_value)
      {
        atomicAdd(&value, N);
        return false;
      }
    return true;
  }

  // ##############################################################################################################################################
  //
  template <typename Desc,typename T>
  inline void BulkSemaphore::wait(const Desc& d,int N, uint32_t number_pages_on_chunk,
                                           T allocationFunction)
  {
    enum class Mode
      {
        AllocateChunk, AllocatePage, Reserve, Invalid
      };
    auto mode{ Mode::Invalid };
    int loop=0;
    while(true)
      {
        if(getCount() - N >= 0)
          {
            // Try to decrement global count first
            uint32_t atomic_ret_val = atomicAdd(&value, Ouro::create2Complement(N)) & highest_value_mask;
            if((atomic_ret_val - N) >= null_value)
              {
                return;
              }
            // Increment count again
            atomicAdd(&value, N);
          }	

        BulkSemaphore old_semaphore_value;
        int expected, reserved, count;
		
        // Read from global
        //BulkSemaphore new_semaphore_value{ atomicAdd(&value,0) };
        BulkSemaphore new_semaphore_value{ value };
        do
          {
            old_semaphore_value = new_semaphore_value;
            new_semaphore_value.getValues(count, expected, reserved);

            if ((count + expected - reserved) < N)
              {
                // We are the one to allocate
                expected += number_pages_on_chunk;
                mode = { Mode::AllocateChunk };
              }
            else if (count >= N)
              {
                // We have enough resources available to satisfy this request
                count -= N;
                mode = { Mode::AllocatePage };
              }
            else
              {
                // Not enough here, let's reserve some stuff and wait
                reserved += N;
                mode = { Mode::Reserve };
              }

            new_semaphore_value.createValueInternal(count, expected, reserved);
            // Try to set new value
          } while ((new_semaphore_value.value = atomicCAS(&value, old_semaphore_value.value, new_semaphore_value.value))
                   != old_semaphore_value.value);

        switch (mode)
          {
          case Mode::AllocatePage: return;
          case Mode::AllocateChunk: allocationFunction(); break;
          }
        
//        // serialise the allocation function within a subgroup
//        auto sg=d.item.get_sub_group();
//        //for (int i=0; i<sg.get_local_linear_range(); ++i)
//        //  if (i==sg.get_local_linear_id() && mode == Mode::AllocateChunk)
//        //    allocationFunction();
//        if (mode == Mode::AllocateChunk) allocationFunction();
//        
//        // ##############################################
//        // Return if chunk allocation or page allocation
//        if (mode == Mode::AllocatePage)
//          return;

        //__syncwarp();
        //sycl::group_barrier(sg);
//        if(mode == Mode::Reserve)
//          {
//            // ##############################################
//            // Wait on our resource
//            unsigned int counter {0U};
//            do
//              {
//                // Yield for some time
//                Ouro::sleep(++counter);
//
//                // Read from global
//                read(new_semaphore_value);
//                new_semaphore_value.getValues(count, expected, reserved);
//                if (counter>10000)
//                  {
//                    d.out<<"Timed out waiting for resources\n";
//                    break;
//                  }
//              } while ((count < N) && (reserved < (count + expected)));
//
//            // ##############################################
//            // Reduce reserved count
//            atomicAdd(&value, create64BitSubAdder_reserved(N));
//          }
        
//        if (loop++>100)
//          {
//            printf("Timed out in allocation loop\n");
//            return;
//          }
      }
  }

  // ##############################################################################################################################################
  //
  inline int BulkSemaphore::signalExpected(unsigned long long N)
  {
    return static_cast<int>(atomicAdd(&value, N + create64BitSubAdder_expected(N)) & highest_value_mask) - null_value;
  }

  // ##############################################################################################################################################
  //
  inline int BulkSemaphore::signal(unsigned long long N)
  {
    return static_cast<int>(atomicAdd(&value, N) & highest_value_mask) - null_value;
  }
}
