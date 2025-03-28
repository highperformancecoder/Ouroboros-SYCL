#pragma once

#include <sycl/sycl.hpp>
#include "Ouroboros_impl.dp.hpp"

namespace Ouro
{
  // ##############################################################################################################################################
  //
  template <typename Desc,typename MemoryManagerType>
  void d_cleanChunks(const Desc& d, MemoryManagerType* memory_manager, unsigned int offset)
  {
    using ChunkType = typename MemoryManagerType::ChunkBase;

    auto group=d.item.get_group();
    for (int chunk=group.get_group_linear_id(); chunk<memory_manager->memory.maxChunks;
         chunk+=group.get_group_linear_range())
      {
        index_t* chunk_data = reinterpret_cast<index_t *>
          (ChunkType::getMemoryAccess(memory_manager->memory.d_data, chunk + offset));
        sycl::group_barrier(group); // TODO - this shouldn't be needed.
        for (int i = group.get_local_linear_id();
             i < ChunkType::size()/sizeof(index_t);
             i += group.get_local_linear_range())
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
  template <typename Desc,typename OUROBOROS>
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
  template <typename Desc>
  inline void
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
  void Ouroboros<OUROBOROS, OUROBOROSES...>::initialize(sycl::queue& syclQueue, sycl::usm::alloc kind, size_t instantiation_size, size_t additionalSizeBeginning, size_t additionalSizeEnd)
  {
    // Initialize memory, then call initialize on all instances
    if (initialized)
      return;
	
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
      {
        memory.d_memory=sycl::malloc<memory_t>(memory.allocationSize, syclQueue, kind);
        memory.context=syclQueue.get_context();
      }
    if (!memory.d_memory)
      {
        std::cout<<"memory allocation failed"<<std::endl;
        abort();
      }
        
    memory.d_data = memory.d_memory + size_();
    memory.d_data_end = memory.d_memory + memory.allocationSize;

    // Put Memory Manager on Device
    updateMemoryManagerDevice(syclQueue,*this);

    auto manager=reinterpret_cast<MyType*>(memory.d_memory);
    syclQueue.single_task([=]() {
      d_setupMemoryPointers<MyType>(manager);
    }).wait_and_throw();

    // Update pointers on host
    updateMemoryManagerHost(syclQueue,*this);

    // Place chunk locator
    memory.d_data_end -= chunk_locator_size;
    memory.chunk_locator.d_chunk_flags = reinterpret_cast<int*>(memory.d_data_end);

    // Lets distribute this pointer to the memory managers
    initMemoryManagers();

    // Lets update the device again to that all the info is there as well
    updateMemoryManagerDevice(syclQueue,*this);

    int block_size = 256;
    auto dev=syclQueue.get_device();
    int grid_size=dev.get_info<sycl::info::device::max_compute_units>()*dev.get_info<sycl::info::device::max_work_group_size>();
    if(totalNumberVirtualQueues())
      {
        // Clean all chunks
        syclQueue.submit([&](auto& h) {
          sycl::stream out(10000,1000,h);
          h.parallel_for(sycl::nd_range<1>(grid_size, block_size),
                         [=](sycl::nd_item<1> item) {
                           Ouro::SyclDesc<1,sycl::stream> d{item,out};
                           d_cleanChunks(d,manager, 0);
                         });
        }).wait_and_throw();
      }

    syclQueue.submit([&](auto& h) {
      sycl::stream out(1000000,1000,h);
      h.parallel_for(sycl::nd_range<1>(grid_size, block_size),
                     [=](sycl::nd_item<1> item) {
                       Ouro::SyclDesc<1,sycl::stream> d{item,out};
                       d_initializeOuroborosQueues(d,manager);
                     });
    }).wait_and_throw();

    updateMemoryManagerHost(syclQueue,*this);

    initialized = true;
  }
}
