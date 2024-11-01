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
void d_testAllocation(MemoryManagerType* mm, int** verification_ptr, int num_allocations, int allocation_size,
                      const sycl::nd_item<1> &item_ct1)
{
        int tid = item_ct1.get_local_id(0) +
                  item_ct1.get_group(0) * item_ct1.get_local_range(0);
        if(tid >= num_allocations)
		return;

	verification_ptr[tid] = reinterpret_cast<int*>(mm->malloc(allocation_size,item_ct1));
}

void d_testWriteToMemory(int** verification_ptr, int num_allocations, int allocation_size,
                         const sycl::nd_item<3> &item_ct1)
{
        int tid = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if(tid >= num_allocations)
		return;
	
	auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		ptr[i] = tid;
	}
}

void d_testReadFromMemory(int** verification_ptr, int num_allocations, int allocation_size,
                          const sycl::nd_item<3> &item_ct1,
                          const sycl::stream &stream_ct1)
{
        int tid = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if(tid >= num_allocations)
		return;

        if (item_ct1.get_local_id(2) == 0 && item_ct1.get_group(2) == 0)
                stream_ct1 << "Test Read!\n";

        auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		if(ptr[i] != tid)
		{
                        /*
                        DPCT1015:0: Output needs adjustment.
                        */
                        stream_ct1 << "%d - %d | We got a wrong value here! %d vs %d\n";
                        return;
		}
	}
}

template <typename MemoryManagerType>
void d_testFree(MemoryManagerType* mm, int** verification_ptr, int num_allocations,
                const sycl::nd_item<1> &item_ct1)
{
        int tid = item_ct1.get_local_id(0) +
                  item_ct1.get_group(0) * item_ct1.get_local_range(0);
        if(tid >= num_allocations)
		return;

	mm->free(verification_ptr[tid]);
}

int main(int argc, char* argv[])
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.in_order_queue();
        std::cout << "Usage: num_allocations allocation_size_in_bytes\n";
	int num_allocations{10000};
	int allocation_size_byte{16};
	int num_iterations {10};
	if(argc >= 2)
	{
		num_allocations = atoi(argv[1]);
		if(argc >= 3)
		{
			allocation_size_byte = atoi(argv[2]);
		}
	}
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

	size_t instantitation_size = 8192ULL * 1024ULL * 1024ULL;
	MemoryManagerType memory_manager;
	memory_manager.initialize(q_ct1, instantitation_size);

	int** d_memory{nullptr};
        HANDLE_ERROR(DPCT_CHECK_ERROR(
            d_memory = sycl::malloc_device<int *>(num_allocations, q_ct1)));

        int blockSize {256};
	int gridSize {Ouro::divup(num_allocations, blockSize)};
	float timing_allocation{0.0f};
	float timing_free{0.0f};
        dpct::event_ptr start, end;
        for(auto i = 0; i < num_iterations; ++i)
	{
		start_clock(start, end);
                q_ct1.parallel_for(sycl::nd_range<1>(gridSize*blockSize, blockSize), [=](const sycl::nd_item<1>& item) {
                  d_testAllocation <MemoryManagerType>(memory_manager.getDeviceMemoryManager(), d_memory, num_allocations, allocation_size_byte, item);
                  });
		timing_allocation += end_clock(start, end);

                HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));

                /*
                DPCT1049:1: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                q_ct1.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize) *
                                          sycl::range<3>(1, 1, blockSize),
                                      sycl::range<3>(1, 1, blockSize)),
                    [=](sycl::nd_item<3> item_ct1) {
                            d_testWriteToMemory(d_memory, num_allocations,
                                                allocation_size_byte, item_ct1);
                    });

                HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));

                /*
                DPCT1049:2: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                q_ct1.submit([&](sycl::handler &cgh) {
                        sycl::stream stream_ct1(64 * 1024, 80, cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, gridSize) *
                                    sycl::range<3>(1, 1, blockSize),
                                sycl::range<3>(1, 1, blockSize)),
                            [=](sycl::nd_item<3> item_ct1) {
                                    d_testReadFromMemory(d_memory,
                                                         num_allocations,
                                                         allocation_size_byte,
                                                         item_ct1, stream_ct1);
                            });
                });

                HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));

                start_clock(start, end);
                q_ct1.parallel_for(sycl::nd_range<1>(gridSize*blockSize, blockSize), [=](const sycl::nd_item<1> item) {
                  d_testFree <MemoryManagerType>(memory_manager.getDeviceMemoryManager(), d_memory, num_allocations, item);
                });
		timing_free += end_clock(start, end);

                HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
        }
	timing_allocation /= num_iterations;
	timing_free /= num_iterations;

	std::cout << "Timing Allocation: " << timing_allocation << "ms" << std::endl;
	std::cout << "Timing       Free: " << timing_free << "ms" << std::endl;

	std::cout << "Testcase DONE!\n";
	
	return 0;
}
