#include <sycl/sycl.hpp>
#include <iostream>

#define DPCT_COMPATIBILITY_TEMP 600

#include "device/Ouroboros_impl.dp.hpp"
#include "device/MemoryInitialization.dp.hpp"
#include "InstanceDefinitions.dp.hpp"
#include "Utility.dp.hpp"

#define TEST_MULTI

template <typename MemoryManagerType>
void d_testAllocation(Ouro::ThreadAllocator<MemoryManagerType>& mm, int** verification_ptr, int num_allocations, int allocation_size)
{
  int tid=mm.desc().item.get_global_linear_id();
  if(tid >= num_allocations)
    return;
  
  verification_ptr[tid] = reinterpret_cast<int*>(mm.malloc(allocation_size));
}

void d_testWriteToMemory(const Ouro::SyclDesc<1,sycl::stream>& d, int** verification_ptr, int num_allocations, int allocation_size)
{
  int tid = d.item.get_global_linear_id();
  if(tid >= num_allocations)
    return;
	
  auto ptr = verification_ptr[tid];

  for(auto i = 0; ptr&&i < (allocation_size / sizeof(int)); ++i)
    {
      ptr[i] = tid;
    }
}

void d_testReadFromMemory(const Ouro::SyclDesc<1,sycl::stream>& d, int** verification_ptr, int num_allocations, int allocation_size)
{
  int tid = d.item.get_global_linear_id();
  if(tid >= num_allocations)
    return;

  if (d.item.get_local_id(0) == 0 && d.item.get_group(0) == 0)
    d.out << "Test Read!\n";

  auto ptr = verification_ptr[tid];

  for(auto i = 0; ptr&&i < (allocation_size / sizeof(int)); ++i)
    {
      if(ptr[i] != tid)
        {
          d.out << d.item.get_local_linear_id()<<" - "<<d.item.get_group_linear_id()<<" | We got a wrong value here! "<<ptr[i]<<" vs "<<tid<<sycl::endl;
          return;
        }
    }
}

template <typename MemoryManagerType>
void d_testFree(Ouro::ThreadAllocator<MemoryManagerType>& mm, int** verification_ptr, int num_allocations)
{
  int tid = mm.desc().item.get_global_linear_id();
  if(tid >= num_allocations)
    return;
  
  mm.free(verification_ptr[tid]);
}

int main(int argc, char* argv[])
{
  sycl::queue q_ct1({sycl::property::queue::enable_profiling()});
  std::cout << "Usage: num_allocations allocation_size_in_bytes\n";
  int num_allocations{8192};
  int allocation_size_byte{16};
  int num_iterations=10;
  int blockSize=256;
  if(argc >= 2)
    {
      num_allocations = atoi(argv[1]);
      if(argc >= 3)
        {
          allocation_size_byte = atoi(argv[2]);
        }
    }
  // num_allocations needs to be a multiple of blocksize for certain SYCL devices.
  num_allocations=(num_allocations/blockSize)*blockSize;
  allocation_size_byte = Ouro::alignment(allocation_size_byte, sizeof(int));
  std::cout << "Number of Allocations: " << num_allocations << " | Allocation Size: " << allocation_size_byte << " | Iterations: " << num_iterations << std::endl;
  std::cout<<"Running on "<<q_ct1.get_device().get_info<sycl::info::device::name>()<<std::endl;
  
#ifdef TEST_PAGES

#ifdef TEST_VIRTUALARRAY
  std::cout << "Testing page-based memory manager - Virtualized Array!\n";
#ifndef TEST_MULTI
  using MemoryManagerType = Ouro::OuroVAPQ;
#else
  using MemoryManagerType = Ouro::MultiOuroVAPQ;
#endif
#elif TEST_VIRTUALLIST
  std::cout << "Testing page-based memory manager - Virtualized List!\n";
#ifndef TEST_MULTI
  using MemoryManagerType = Ouro::OuroVLPQ;
#else
  using MemoryManagerType = Ouro::MultiOuroVLPQ;
#endif
#else
  std::cout << "Testing page-based memory manager - Standard!\n";
#ifndef TEST_MULTI
  using MemoryManagerType = Ouro::OuroPQ;
#else
  using MemoryManagerType = Ouro::MultiOuroPQ;
#endif
#endif

#elif TEST_CHUNKS

#ifdef TEST_VIRTUALARRAY
  std::cout << "Testing chunk-based memory manager - Virtualized Array!\n";
#ifndef TEST_MULTI
  using MemoryManagerType = Ouro::OuroVACQ;
#else
  using MemoryManagerType = Ouro::MultiOuroVACQ;
#endif
#elif TEST_VIRTUALLIST
  std::cout << "Testing chunk-based memory manager - Virtualized List!\n";
#ifndef TEST_MULTI
  using MemoryManagerType = Ouro::OuroVLCQ;
#else
  using MemoryManagerType = Ouro::MultiOuroVLCQ;
#endif
#else
  std::cout << "Testing chunk-based memory manager - Standard!\n";
#ifndef TEST_MULTI
  using MemoryManagerType = Ouro::OuroCQ;
#else
  using MemoryManagerType = Ouro::MultiOuroCQ;
#endif
#endif

#endif

  size_t instantitation_size = /*8192ULL*/512ULL * 1024ULL * 1024ULL;
          
  MemoryManagerType memory_manager;
  memory_manager.initialize(q_ct1, sycl::usm::alloc::device, instantitation_size);

  int** d_memory = sycl::malloc_device<int *>(num_allocations, q_ct1);

  float timing_allocation{0.0f};
  float timing_free{0.0f};
  auto deviceMemMgr=memory_manager.getDeviceMemoryManager();
  for(auto i = 0; i < num_iterations; ++i)
    {
      std::cout<<"alloc "<<i<<std::endl;
      auto ev=q_ct1.submit([&](auto& h) {
        sycl::stream out(1000000,1000,h);
        h.parallel_for(sycl::nd_range<1>(num_allocations, blockSize), [=](const sycl::nd_item<1>& item) {
          Ouro::ThreadAllocator<MemoryManagerType> m(item,out,*deviceMemMgr);
          d_testAllocation(m, d_memory, num_allocations, allocation_size_byte);
        });
      });
      ev.wait_and_throw();
      timing_allocation += float(1e-6*(ev.get_profiling_info<sycl::info::event_profiling::command_end>()-
                                       ev.get_profiling_info<sycl::info::event_profiling::command_start>()));

      std::cout<<"write "<<i<<std::endl;
      q_ct1.submit([&](auto& h) {
        sycl::stream out(1000000,1000,h);
        h.parallel_for(
                       sycl::nd_range<1>(num_allocations, blockSize),
                       [=](sycl::nd_item<1> item) {
                         Ouro::SyclDesc<1,sycl::stream> d{item,out};
                         d_testWriteToMemory(d, d_memory, num_allocations,
                                             allocation_size_byte);
                       });
      }).wait_and_throw();
                  
      std::cout<<"read "<<i<<std::endl;
      q_ct1.submit([&](auto& h) {
        sycl::stream out(1000000,1000,h);
        h.parallel_for(
                       sycl::nd_range<1>(num_allocations, blockSize),
                       [=](sycl::nd_item<1> item) {
                         Ouro::SyclDesc<1,sycl::stream> d{item,out};
                         d_testReadFromMemory(d,d_memory,
                                              num_allocations,
                                              allocation_size_byte);
                       });
      }).wait_and_throw();

      std::cout<<"free "<<i<<std::endl;
      ev=q_ct1.submit([&](auto& h) {
        sycl::stream out(1000000,1000,h);
        h.parallel_for(sycl::nd_range<1>(num_allocations, blockSize), [=](const sycl::nd_item<1> item) {
          Ouro::ThreadAllocator<MemoryManagerType> m(item,out,*deviceMemMgr);
          d_testFree(m, d_memory, num_allocations);
        });
      });
      ev.wait_and_throw();
      timing_free += float(1e-6*(ev.get_profiling_info<sycl::info::event_profiling::command_end>()-
                                       ev.get_profiling_info<sycl::info::event_profiling::command_start>()));
    }
  timing_allocation /= num_iterations;
  timing_free /= num_iterations;

  std::cout << "Timing Allocation: " << timing_allocation << "ms" << std::endl;
  std::cout << "Timing       Free: " << timing_free << "ms" << std::endl;

  std::cout << "Testcase DONE!\n";

  sycl::free(d_memory, q_ct1);
  return 0;
}
