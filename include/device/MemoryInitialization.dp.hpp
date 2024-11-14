#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Ouroboros_impl.dp.hpp"

// ##############################################################################################################################################
//
void printCompute(const sycl::nd_item<3> &item_ct1,
                  const sycl::stream &stream_ct1)
{
        int tid = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if(tid >= 1)
		return;
#if (DPCT_COMPATIBILITY_TEMP >= 700)
                        stream_ct1 << "ASYNC - COMPUTE MODE";
#else
                        //			printf("SYNC - COMPUTE MODE");
		#endif
}

// ##############################################################################################################################################
//
template <typename MemoryManagerType>
void d_cleanChunks(MemoryManagerType* memory_manager, unsigned int offset,
                   sycl::nd_item<1> item_ct1)
{
	using ChunkType = typename MemoryManagerType::ChunkBase;
        index_t* chunk_data;
 
        if (item_ct1.get_local_id(0) == 0)
        {
                 chunk_data =
                     reinterpret_cast<index_t *>(ChunkType::getMemoryAccess(
                         memory_manager->memory.d_data,
                         item_ct1.get_group(2) + offset));
        }

        /*
        DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        for (int i = item_ct1.get_local_id(0);
             i < (MemoryManagerType::ChunkBase::size_ +
                  MemoryManagerType::ChunkBase::meta_data_size_);
             i += item_ct1.get_local_range(0))
        {
		chunk_data[i] = DeletionMarker<index_t>::val;
	}
}

// ##############################################################################################################################################
//
template <typename MemoryManagerType>
void initNew(MemoryManagerType& memory_manager, memory_t** d_data_end)
{
	// Place Chunk Queue
	*d_data_end -= MemoryManagerType::chunk_queue_size_;
	memory_manager.d_chunk_reuse_queue.queue_ = reinterpret_cast<decltype(memory_manager.d_chunk_reuse_queue.queue_)>(*d_data_end);
	memory_manager.d_chunk_reuse_queue.size_ = chunk_queue_size;

	// Place Page Queues
	for (auto i = 0; i < MemoryManagerType::NumberQueues_; ++i)
	{
		*d_data_end -= MemoryManagerType::page_queue_size_;
		memory_manager.d_storage_reuse_queue[i].queue_ = reinterpret_cast<index_t*>(*d_data_end);
		memory_manager.d_storage_reuse_queue[i].queue_index_ = i;
		memory_manager.d_storage_reuse_queue[i].page_size_ = MemoryManagerType::SmallestPageSize_ << i;
	}
}

// ##############################################################################################################################################
//
template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
void OuroborosChunks<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::initializeNew(memory_t** d_data_end)
{
	initNew<OuroborosChunks<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>>(*this, d_data_end);
}

// ##############################################################################################################################################
//
template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
void OuroborosPages<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>::initializeNew(memory_t** d_data_end)
{
	initNew<OuroborosPages<QUEUE_TYPE, CHUNK_BASE, SMALLEST_SIZE, NUMBER_QUEUES>>(*this, d_data_end);
}

// ##############################################################################################################################################
//
template <typename OUROBOROS>
void d_setupMemoryPointers(OUROBOROS* ouroboros)
{
	// Template-recursive give all memory managers the same pointer
	ouroboros->setMemoryPointer();
}

// ##############################################################################################################################################
//
template <typename OUROBOROS>
void d_initializeOuroborosQueues(const Desc& d,OUROBOROS* ouroboros)
{
	// Template-recursive to initialize queues
  ouroboros->memory.chunk_locator.init(d,ouroboros->memory.maxChunks);
	IndexQueue* d_base_chunk_reuse{nullptr};
	ouroboros->initQueues(d,d_base_chunk_reuse);
}

// ##############################################################################################################################################
//
template <class OUROBOROS, class... OUROBOROSES>
__dpct_inline__ void
Ouroboros<OUROBOROS, OUROBOROSES...>::initQueues(const Desc& d, IndexQueue *d_base_chunk_reuse)
{
	// --------------------------------------------------------
	// Init queues
  memory_manager.d_chunk_reuse_queue.init(d);
	if(d_base_chunk_reuse == nullptr)
		d_base_chunk_reuse = &(memory_manager.d_chunk_reuse_queue);
	memory_manager.d_base_chunk_reuse_queue = d_base_chunk_reuse;
	#pragma unroll
	for (auto i = 0; i < ConcreteOuroboros::NumberQueues_; ++i)
	{
          memory_manager.d_storage_reuse_queue[i].init(d,&memory_manager);
	}

	// Init next queues
	next_memory_manager.initQueues(d,d_base_chunk_reuse);
}

// ##############################################################################################################################################
//
template<class OUROBOROS, class... OUROBOROSES>
void Ouroboros<OUROBOROS, OUROBOROSES...>::initialize(sycl::queue& syclQueue, size_t instantiation_size, size_t additionalSizeBeginning, size_t additionalSizeEnd)
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
        // Initialize memory, then call initialize on all instances
	if (initialized)
		return;
	
//	if(!cuda_initialized)
//	{
//		cudaDeviceSetLimit(cudaLimitMallocHeapSize, cuda_heap_size);
//		cuda_initialized = true;
//	}
	size_t size;
        /*
        DPCT1029:1: SYCL currently does not support getting device resource
        limits. The output parameter(s) are set to 0.
        */
        *&size = 0;
//        if(printDebug)
//		printf("Heap Size: ~%llu MB\n", size / (1024 * 1024));
//
//	if(printDebug)
//	{
//		printf("%s##\n####\n##\n---", break_line_green_s);
//                dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
//                        sycl::stream stream_ct1(64 * 1024, 80, cgh);
//
//                        cgh.parallel_for(
//                            sycl::nd_range<3>(sycl::range<3>(1, 1, 32),
//                                              sycl::range<3>(1, 1, 32)),
//                            [=](sycl::nd_item<3> item_ct1) {
//                                    printCompute(item_ct1, stream_ct1);
//                            });
//                });
//                dev_ct1.queues_wait_and_throw();
//#ifdef TEST_VIRTUALIZED
//		printf(" - VIRTUALIZED ARRAY-BASED");
//#elif TEST_VIRTUALIZED_LINKED
//		printf(" - VIRTUALIZED LINKED-LIST-BASED");
//#else
//		printf(" - STANDARD");
//#endif
//		printf("  ---\n##\n####\n##\n%s", break_line_green_e);
//	}

	// Get total size from all Memory Managers
	auto total_memory_manager_size = totalMemoryManagerSize();

	// Align both the required size and total size to the chunk base size
	auto total_required_size = Ouro::alignment<size_t>(size_() + total_memory_manager_size + additionalSizeBeginning + additionalSizeEnd, ChunkBase::size());
	auto difference = Ouro::alignment<size_t>(instantiation_size, ChunkBase::size()) - total_required_size;
	memory.maxChunks = difference / ChunkBase::size();
	memory.adjacencysize = Ouro::alignment<uint64_t>(memory.maxChunks * ChunkBase::size());
	size_t chunk_locator_size = Ouro::alignment<size_t>(ChunkLocator::size(memory.maxChunks), ChunkBase::size());
	memory.allocationSize = Ouro::alignment<uint64_t>(total_required_size + memory.adjacencysize + chunk_locator_size, ChunkBase::size());
	memory.additionalSizeBeginning = additionalSizeBeginning;
	memory.additionalSizeEnd = additionalSizeEnd;

	// Allocate memory
	if (!memory.d_memory)
          //cudaMalloc(reinterpret_cast<void**>(&memory.d_memory), memory.allocationSize);
          memory.d_memory=sycl::malloc_device<memory_t>(memory.allocationSize, syclQueue);
          

	memory.d_data = memory.d_memory + size_();
	memory.d_data_end = memory.d_memory + memory.allocationSize;

	// Put Memory Manager on Device
	updateMemoryManagerDevice(*this);

        syclQueue.single_task([this]() {
          d_setupMemoryPointers<MyType>(reinterpret_cast<MyType*>(memory.d_memory));
        });

        HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));

        // Update pointers on host
	updateMemoryManagerHost(*this);

	// Place chunk locator
	memory.d_data_end -= chunk_locator_size;
	memory.chunk_locator.d_chunk_flags = reinterpret_cast<int*>(memory.d_data_end);

	// Lets distribute this pointer to the memory managers
	initMemoryManagers();

	// Lets update the device again to that all the info is there as well
	updateMemoryManagerDevice(*this);

	int block_size = 256;
	int grid_size = memory.maxChunks;
	if(totalNumberVirtualQueues())
	{
          // Clean all chunks
          syclQueue.parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size), [this](sycl::nd_item<1> item) {
            d_cleanChunks<MyType>(reinterpret_cast<MyType*>(memory.d_memory), 0, item);
          });
	}

        HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));

        block_size = 256;
	//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size, d_initializeOuroborosQueues<MyType>, block_size, 0U);
        grid_size=dev_ct1.get_info<sycl::info::device::max_compute_units>()*dev_ct1.get_info<sycl::info::device::max_work_group_size>()/block_size; // TODO - this is my guess...
//	int num_sm_per_device{0};
//        num_sm_per_device = dpct::get_device(0).get_max_compute_units();
//        grid_size *= num_sm_per_device;
        syclQueue.submit([&](auto& h) {
          sycl::stream out(1000000,1000,h);
          h.parallel_for(sycl::nd_range<1>(grid_size*block_size, block_size),
                         [=,this](sycl::nd_item<1> item) {
                           Desc d{item,out};
                           d_initializeOuroborosQueues<MyType>(d,reinterpret_cast<MyType*>(memory.d_memory));
                         });
        });

        HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));

        updateMemoryManagerHost(*this);

	initialized = true;

        HANDLE_ERROR(DPCT_CHECK_ERROR(dev_ct1.queues_wait_and_throw()));
}
