#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "BulkSemaphore.dp.hpp"

namespace Ouro
{
  // ##############################################################################################################################################
  //
  __dpct_inline__ bool BulkSemaphore::tryReduce(int N)
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
  __dpct_inline__ void BulkSemaphore::wait(const Desc& d,int N, uint32_t number_pages_on_chunk,
                                           T allocationFunction)
  //#if (DPCT_COMPATIBILITY_TEMP < 700)
#if 1
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

        // serialise the allocation function within a subgroup
        auto sg=d.item.get_sub_group();
        for (int i=0; i<sg.get_local_linear_range(); ++i)
          if (i==sg.get_local_linear_id() && mode == Mode::AllocateChunk)
            allocationFunction();

        //__syncwarp();
        sycl::group_barrier(sg);
        // ##############################################
        // Return if chunk allocation or page allocation
        if (mode == Mode::AllocatePage)
          return;

        if(mode == Mode::Reserve)
          {
            // ##############################################
            // Wait on our resource
            unsigned int counter {0U};
            do
              {
                // Yield for some time
                Ouro::sleep(++counter);

                // Read from global
                read(new_semaphore_value);
                new_semaphore_value.getValues(count, expected, reserved);
                if (counter>4000)
                  d.out<<d.item.get_global_linear_id()<<" count="<<count<<" expected="<<expected<<" reserved="<<reserved<<" N="<<N<<" counter="<<counter<<sycl::endl;
                if (counter>4096)
                  {
                    d.out<<"Timed out waiting for resources\n";
                    break;
                  }
              } while ((count < N) && (reserved < (count + expected)));

            // ##############################################
            // Reduce reserved count
            atomicAdd(&value, create64BitSubAdder_reserved(N));
          }
        if (loop>90)
          d.out<<d.item.get_global_linear_id()<<" mode="<<int(mode)<<" count="<<count<<" loop="<<loop<<sycl::endl;
        if (loop++>100)
          {
            d.out<<"Timed out in allocation loop\n";
            return;
          }
      }
  }
#else
  // ##############################################################################################################################################
  //
{
  enum class Mode
    {
      AllocateChunk, AllocatePage, Reserve, Invalid
    };
  auto mode{ Mode::Invalid };
  /*
    DPCT1086:0: __activemask() is migrated to 0xffffffff. You may need to adjust
    the code.
  */
  auto sg=d.item.get_sub_group();
//  int mask = dpct::match_any_over_sub_group(sg, 0xffffffff, number_pages_on_chunk);
//  int leader = dpct::ffs<int>(mask) - 1; // Select leader
//
//  int leader_mask = __match_any_sync(0xffffffff, Ouro::lane_id() == leader);
  if(sg.leader())
    {
      int num = sg.get_local_linear_range()*N; // How much should our allocator allocate?
      while(true)
        {
          if(getCount() - static_cast<int>(num) >= 0)
            {
              uint32_t atomic_ret_val = atomicAdd(&value, Ouro::create2Complement(num)) & highest_value_mask; // Try to decrement global count first
              if((atomic_ret_val - num) >= null_value)
                {
                  break;
                }

              /*
                DPCT1078:1: Consider replacing
                memory_order::acq_rel with memory_order::seq_cst
                for correctness if strong memory order
                restrictions are needed.
              */
              sycl::atomic_fence(
                                 sycl::memory_order::seq_cst,
                                 sycl::memory_scope::work_group);

              // Increment count again
              atomicAdd(&value, num);
            }
			
          BulkSemaphore old_semaphore_value;
          int expected, reserved, count;
          // Read from global
          BulkSemaphore new_semaphore_value{ Ouro::Atomic<unsigned long long>(value) };
          do
            {
              old_semaphore_value = new_semaphore_value;
              new_semaphore_value.getValues(count, expected, reserved);

              if ((count + expected - reserved) < num)
                {
                  // We are the one to allocate
                  expected += number_pages_on_chunk;
                  mode = { Mode::AllocateChunk };
                }
              else if (count >= num)
                {
                  // TODO: Can we maybe try this with an atomic?
                  // We have enough resources available to satisfy this request
                  count -= num;
                  mode = { Mode::AllocatePage };
                }
              else
                {
                  // Not enough here, let's reserve some stuff and wait
                  reserved += num;
                  mode = { Mode::Reserve };
                }

              new_semaphore_value.createValueInternal(count, expected, reserved);
              // Try to set new value
            }
          while ((new_semaphore_value.value = atomicCAS(&value, old_semaphore_value.value, new_semaphore_value.value))
                 != old_semaphore_value.value);
          if (mode == Mode::AllocatePage)
            break;

          if(mode == Mode::AllocateChunk)
            {
              allocationFunction();
            }
          // TODO: Not needed??
          //sycl::group_barrier(sg);
          // __threadfence_block();

          if(mode == Mode::Reserve)
            {
              // ##############################################
              // Wait on our resource
              unsigned int counter {0U};
              do
                {
                  // Yield for some time
                  Ouro::sleep(++counter);

                  // Read from global
                  read(new_semaphore_value);
                  new_semaphore_value.getValues(count, expected, reserved);
                } 
              while ((count < num) && (reserved < (count + expected)));

              // ##############################################
              // Reduce reserved count
              atomicAdd(&value, create64BitSubAdder_reserved(num));
            }
        }
    }
  // Gather all threads around -> wait for the leader to finish
  sycl::group_barrier(sg);
}
#endif

  // ##############################################################################################################################################
  //
  __dpct_inline__ int BulkSemaphore::signalExpected(unsigned long long N)
  {
    return static_cast<int>(atomicAdd(&value, N + create64BitSubAdder_expected(N)) & highest_value_mask) - null_value;
  }

  // ##############################################################################################################################################
  //
  __dpct_inline__ int BulkSemaphore::signal(unsigned long long N)
  {
    return static_cast<int>(atomicAdd(&value, N) & highest_value_mask) - null_value;
  }
}
