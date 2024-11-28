#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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

void d_testWriteToMemory(const Desc& d, int** verification_ptr, int num_allocations, int allocation_size)
{
  int tid = d.item.get_global_linear_id();
  if(tid >= num_allocations)
    return;
	
  auto ptr = verification_ptr[tid];

  for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
    {
      ptr[i] = tid;
    }
}

void d_testReadFromMemory(const Desc& d, int** verification_ptr, int num_allocations, int allocation_size)
{
  int tid = d.item.get_global_linear_id();
  if(tid >= num_allocations)
    return;

  if (d.item.get_local_id(0) == 0 && d.item.get_group(0) == 0)
    d.out << "Test Read!\n";

  auto ptr = verification_ptr[tid];

  for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
    {
      if(ptr[i] != tid)
        {
          /*
            DPCT1015:0: Output needs adjustment.
          */
          d.out << d.item.get_local_id(0)<<" - "<<d.item.get_group(0)<<" | We got a wrong value here! "<<ptr[i]<<" vs "<<tid<<sycl::endl;
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
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.in_order_queue();
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

	#ifdef TEST_PAGES

	#ifdef TEST_VIRTUALARRAY
	std::cout << "Testing page-based memory manager - Virtualized Array!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVAPQ;
	#else
	using MemoryManagerType = MultiOuroVAPQ;
	#endif
	#elif TEST_VIRTUALLIST
	std::cout << "Testing page-based memory manager - Virtualized List!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVLPQ;
	#else
	using MemoryManagerType = MultiOuroVLPQ;
	#endif
	#else
	std::cout << "Testing page-based memory manager - Standard!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroPQ;
	#else
	using MemoryManagerType = MultiOuroPQ;
	#endif
	#endif

	#elif TEST_CHUNKS

	#ifdef TEST_VIRTUALARRAY
	std::cout << "Testing chunk-based memory manager - Virtualized Array!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVACQ;
	#else
	using MemoryManagerType = MultiOuroVACQ;
	#endif
	#elif TEST_VIRTUALLIST
	std::cout << "Testing chunk-based memory manager - Virtualized List!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroVLCQ;
	#else
	using MemoryManagerType = MultiOuroVLCQ;
	#endif
	#else
	std::cout << "Testing chunk-based memory manager - Standard!\n";
	#ifndef TEST_MULTI
	using MemoryManagerType = OuroCQ;
	#else
	using MemoryManagerType = MultiOuroCQ;
	#endif
	#endif

	#endif

	size_t instantitation_size = /*8192ULL*/512ULL * 1024ULL * 1024ULL;
          
	MemoryManagerType* memory_manager=sycl::malloc_shared<MemoryManagerType>(1, q_ct1);
        new(memory_manager) MemoryManagerType;
	memory_manager->initialize(q_ct1, instantitation_size);

	int** d_memory{nullptr};
        HANDLE_ERROR(DPCT_CHECK_ERROR(
            d_memory = sycl::malloc_device<int *>(num_allocations, q_ct1)));

	float timing_allocation{0.0f};
	float timing_free{0.0f};
        //dpct::event_ptr start, end;
        for(auto i = 0; i < num_iterations; ++i)
	{
          auto start=clock();
          q_ct1.submit([&](auto& h) {
                  sycl::stream out(1000000,1000,h);
                  h.parallel_for(sycl::nd_range<1>(num_allocations, blockSize), [=](const sycl::nd_item<1>& item) {
                    Ouro::ThreadAllocator<MemoryManagerType> m(item,out,*memory_manager->getDeviceMemoryManager());
                    d_testAllocation(m, d_memory, num_allocations, allocation_size_byte);
                  });
                });
                q_ct1.wait();
		timing_allocation += float(clock()-start)/CLOCKS_PER_SEC;

                HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));

                /*
                DPCT1049:1: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                q_ct1.submit([&](auto& h) {
                  sycl::stream out(1000000,1000,h);
                  h.parallel_for(
                                 sycl::nd_range<1>(num_allocations, blockSize),
                                 [=](sycl::nd_item<1> item) {
                                   Desc d{item,out};
                                   d_testWriteToMemory(d, d_memory, num_allocations,
                                                       allocation_size_byte);
                                 });
                });
                  
                HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));

                /*
                DPCT1049:2: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                q_ct1.submit([&](auto& h) {
                  sycl::stream out(1000000,1000,h);
                  h.parallel_for(
                                 sycl::nd_range<1>(num_allocations, blockSize),
                                 [=](sycl::nd_item<1> item) {
                                   Desc d{item,out};
                                   d_testReadFromMemory(d,d_memory,
                                                         num_allocations,
                                                         allocation_size_byte);
                                 });
                });

                HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));

                start=clock();
                q_ct1.submit([&](auto& h) {
                  sycl::stream out(1000000,1000,h);
                  h.parallel_for(sycl::nd_range<1>(num_allocations, blockSize), [=](const sycl::nd_item<1> item) {
                    Ouro::ThreadAllocator<MemoryManagerType> m(item,out,*memory_manager->getDeviceMemoryManager());
                    d_testFree(m, d_memory, num_allocations);
                  });
                });
                q_ct1.wait();
		timing_free += float(clock()-start)/CLOCKS_PER_SEC;

                HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
        }
	timing_allocation /= num_iterations;
	timing_free /= num_iterations;

	std::cout << "Timing Allocation: " << timing_allocation << "s" << std::endl;
	std::cout << "Timing       Free: " << timing_free << "s" << std::endl;

	std::cout << "Testcase DONE!\n";
	
	return 0;
}
