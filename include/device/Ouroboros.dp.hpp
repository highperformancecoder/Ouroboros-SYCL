#pragma once

#include "device/MemoryIndex.dp.hpp"
#include "device/queues/Queues.dp.hpp"
#include "Statistics.h"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Helper.dp.hpp"

namespace Ouro
{
  struct Memory
  {
    Memory(){}
    ~Memory()
    {
      //		if(d_memory != nullptr)
      //			HANDLE_ERROR(cudaFree(d_memory));
    }
    // Data
    memory_t* d_memory{ nullptr };
    memory_t* d_data{ nullptr };
    memory_t* d_data_end{ nullptr };
    index_t next_free_chunk{ 0 };

    size_t maxChunks{ 0 };
    size_t allocationSize{ 0 };
    size_t adjacencysize{ 0 };
    size_t additionalSizeBeginning{0};
    size_t additionalSizeEnd{0};

    ChunkLocator chunk_locator;
  };

  struct OuroborosBase
  {
    // Data
    memory_t* d_memory{ nullptr };
    memory_t* d_data{ nullptr };
    index_t* next_free_chunk{ nullptr };
    size_t maxChunks{ 0 };
    ChunkLocator* chunk_locator{nullptr};

    // Re-Use Queue
    IndexQueue d_chunk_reuse_queue;
    IndexQueue* d_base_chunk_reuse_queue{nullptr};

    // Some statistics in there
    Statistics stats;

    // Error Code
    Ouro::ErrorType error{Ouro::ErrorVal<Ouro::ErrorType, Ouro::ErrorCodes::NO_ERROR>::value};

    bool checkError() { return Ouro::ErrorVal<Ouro::ErrorType, Ouro::ErrorCodes::NO_ERROR>::checkError(error); }
  };

  template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
  struct OuroborosChunks : OuroborosBase
  {
    static constexpr bool ManageChunks{ true };

    static constexpr unsigned int SmallestPageSize_{SMALLEST_SIZE};
    static constexpr unsigned int NumberQueues_{NUMBER_QUEUES};
    static constexpr unsigned int LargestPageSize_{SmallestPageSize_ << (NumberQueues_ - 1)};
    static constexpr unsigned int ChunkSize_{ SmallestPageSize_ << (NumberQueues_ - 1) };
    static constexpr unsigned int ChunkAddFactor_{ChunkSize_ / CHUNK_BASE::size_};
    static constexpr bool s_isBaseOuroboros{ChunkAddFactor_ == 1};

	
    using ChunkBase = CHUNK_BASE;
    using ChunkType = ChunkIndexChunk<ChunkBase, ChunkSize_, SmallestPageSize_>;
    using QueueType = QUEUE_TYPE<ChunkType>;
    using QI = QueueIndex<SmallestPageSize_, ChunkSize_>;

    static constexpr size_t memory_manager_size_() { return Ouro::alignment<uint64_t>(sizeof(OuroborosChunks)); };
    static constexpr size_t chunk_queue_size_{Ouro::alignment<uint64_t>(chunk_queue_size * sizeof(index_t), ChunkBase::size_)};
    static constexpr size_t page_queue_size_{Ouro::alignment<uint64_t>(QueueType::size_ * sizeof(MemoryIndex), ChunkBase::size_)};

    QueueType d_storage_reuse_queue[NumberQueues_];

    static constexpr size_t memoryManagerSize() { return chunk_queue_size_ + (page_queue_size_ * NumberQueues_); }

    void initialize(size_t additionalSizeBeginning = 0, size_t additionalSizeEnd = 0);

    void initializeNew(memory_t** d_data_end);

    void reinitialize(float overallocation_factor);

    __dpct_inline__ void *allocPage(const Desc& d, size_t size);

    __dpct_inline__ void freePage(const Desc& d, MemoryIndex index);

    __dpct_inline__ void initializeQueues();

    __dpct_inline__ void printFreeResources();

    template <bool QUEUECHUNK = false>
    __dpct_inline__ void enqueueChunkForReuse(index_t chunk_index)
    {
      if(!s_isBaseOuroboros && QUEUECHUNK)
        {
          d_base_chunk_reuse_queue->enqueue(chunk_index);
        }
      else
        {
          d_chunk_reuse_queue.enqueue(chunk_index);
        }
    }

    // #################################################################################################
    // Functionality
    template <bool QUEUECHUNK = false>
    __dpct_inline__ bool allocateChunk(index_t &chunk_index)
    {
#ifdef DPCT_COMPATIBILITY_TEMP

      if(statistics_enabled)
        atomicAdd(&stats.chunkAllocationCount, 1);

      if(!s_isBaseOuroboros && QUEUECHUNK)
        {
          if(d_base_chunk_reuse_queue->dequeue(chunk_index))
            {
              return true;
            }
        }
      else
        {
          if(d_chunk_reuse_queue.dequeue(chunk_index))
            {
              return true;
            }
        }

      chunk_index = atomicAdd(next_free_chunk, (QUEUECHUNK ? 1 : ChunkAddFactor_));
      chunk_locator->initChunkIndex(chunk_index);
      return (chunk_index + (QUEUECHUNK ? 1 : ChunkAddFactor_)) < maxChunks;

#endif
    }

    void printQueueStatistics()
    {
      printf("%sPage Queue Fill Rate\n%s", break_line_purple_s, break_line);
      for(auto i = 0; i < NumberQueues_; ++i)
        {
          Ouro::printProgressBar(d_storage_reuse_queue[i].getCount() / static_cast<double>(d_storage_reuse_queue[i].size_));
          Ouro::printProgressBarEnd();
        }
      printf("%s", break_line_purple);
    }
  };

  template <template <class /*CHUNK_TYPE*/> class QUEUE_TYPE, typename CHUNK_BASE, unsigned int SMALLEST_SIZE, unsigned int NUMBER_QUEUES>
  struct OuroborosPages : OuroborosBase
  {
    static constexpr bool ManageChunks{ false };

    static constexpr unsigned int SmallestPageSize_{SMALLEST_SIZE};
    static constexpr unsigned int NumberQueues_{NUMBER_QUEUES};
    static constexpr unsigned int LargestPageSize_{SmallestPageSize_ << (NumberQueues_ - 1)};
    static constexpr unsigned int ChunkSize_{ SmallestPageSize_ << (NumberQueues_ - 1) };
    static constexpr unsigned int ChunkAddFactor_{ChunkSize_ / CHUNK_BASE::size_};
    static constexpr bool s_isBaseOuroboros{ChunkAddFactor_ == 1};

    using ChunkBase = CHUNK_BASE;
    using ChunkType = PageChunk<ChunkBase, ChunkSize_>;
    using QueueType = QUEUE_TYPE<ChunkType>;
    using QI = QueueIndex<SmallestPageSize_, ChunkSize_>;

    static constexpr size_t memory_manager_size_() { return Ouro::alignment<uint64_t>(sizeof(OuroborosPages)); };
    static constexpr size_t chunk_queue_size_{Ouro::alignment<uint64_t>(chunk_queue_size * sizeof(index_t), ChunkBase::size_)};
    static constexpr size_t page_queue_size_{Ouro::alignment<uint64_t>(QueueType::size_ * sizeof(MemoryIndex), ChunkBase::size_)};

    QueueType d_storage_reuse_queue[NumberQueues_];

    static constexpr size_t memoryManagerSize() { return chunk_queue_size_ + (page_queue_size_ * NumberQueues_); }

    void initialize(size_t additionalSizeBeginning = 0, size_t additionalSizeEnd = 0);

    void initializeNew(memory_t** d_data_end);

    void reinitialize(float overallocation_factor);

    __dpct_inline__ void *allocPage(const Desc&,size_t size);

    __dpct_inline__ void freePage(const Desc&,MemoryIndex index);

    template <bool QUEUECHUNK = false>
    __dpct_inline__ void enqueueChunkForReuse(index_t chunk_index)
    {
      if(!s_isBaseOuroboros && QUEUECHUNK)
        {
          d_base_chunk_reuse_queue->enqueue(chunk_index);
        }
      else
        {
          d_chunk_reuse_queue.enqueue(chunk_index);
        }
    }

    // #################################################################################################
    // Functionality
    template <bool QUEUECHUNK = false>
    __dpct_inline__ bool allocateChunk(index_t &chunk_index)
    {
#ifdef DPCT_COMPATIBILITY_TEMP

      if(statistics_enabled)
        atomicAdd(&stats.chunkAllocationCount, 1);

      if(!s_isBaseOuroboros && QUEUECHUNK)
        {
          if(d_base_chunk_reuse_queue->dequeue(chunk_index))
            {
              return true;
            }
        }
      else
        {
          if(d_chunk_reuse_queue.dequeue(chunk_index))
            {
              return true;
            }
        }

      chunk_index = atomicAdd(next_free_chunk, (QUEUECHUNK ? 1 : ChunkAddFactor_));
      // chunk_index = atomicAdd(next_free_chunk, ChunkAddFactor_);
      chunk_locator->initChunkIndex(chunk_index);
      return (chunk_index + (QUEUECHUNK ? 1 : ChunkAddFactor_)) < maxChunks;

#endif
    }

    __dpct_inline__ void initializeQueues();

    void printQueueStatistics()
    {
      printf("%sPage Queue Fill Rate\n%s", break_line_purple_s, break_line);
      for(auto i = 0; i < NumberQueues_; ++i)
        {
          Ouro::printProgressBar(d_storage_reuse_queue[i].getCount() / static_cast<double>(d_storage_reuse_queue[i].size_));
          Ouro::printProgressBarEnd();
        }
      printf("%s", break_line_purple);
    }

    __dpct_inline__ void printFreeResources();
  };

  template<class... OUROBOROSES>
  struct Ouroboros;

  template<class OUROBOROS, class... OUROBOROSES>
  struct Ouroboros<OUROBOROS, OUROBOROSES...>
  {
    using ConcreteOuroboros = OUROBOROS;
    using Next = Ouroboros<OUROBOROSES...>;
    using ChunkBase = typename OUROBOROS::ChunkBase;
    using ChunkType = typename OUROBOROS::ChunkType;
    using MyType = Ouroboros<OUROBOROS, OUROBOROSES...>;
    using QI = typename ConcreteOuroboros::QI;

    static constexpr size_t size_() { return Ouro::alignment<size_t>(sizeof(Ouroboros<OUROBOROS, OUROBOROSES...>), ChunkBase::size_); };

    static constexpr bool checkSizeConstraints() 
    { 
      return 
        (countBitShift(ConcreteOuroboros::ChunkSize_ / ConcreteOuroboros::SmallestPageSize_) <= NUM_BITS_FOR_PAGE) 
        && 
        Next::checkSizeConstraints();
    }
    static_assert(checkSizeConstraints(), "Size Constraints do not match! Check the instantiation parameters.");

    Memory memory;
    ConcreteOuroboros memory_manager;
    Next next_memory_manager;

    bool initialized{false};
    Statistics stats;

    // -----------------------------------------------------------------------------------------------------------
    // Public Interface

    void initialize(sycl::queue& syclQueue, size_t instantiation_size, size_t additionalSizeBeginning = 0, size_t additionalSizeEnd = 0);

    void reinitialize(float overallocation_factor);

    __dpct_inline__ void *malloc(const Desc&,size_t size);

    __dpct_inline__ void free(const Desc&,void *ptr);

    __dpct_inline__ void freePageRecursive(const Desc&,unsigned int page_size,
                                           MemoryIndex index);

    __dpct_inline__ void enqueueInitialChunk(index_t queue_index,
                                             index_t chunk_index,
                                             int available_pages,
                                             index_t pages_per_chunk)
    {
      // TODO: This should later relay to the correct mem man
      memory_manager.d_storage_reuse_queue[queue_index].enqueueInitialChunk(&memory_manager, chunk_index, available_pages, pages_per_chunk);
    }

    MyType* getDeviceMemoryManager() const {return reinterpret_cast<MyType*>(memory.d_memory);}

    // -----------------------------------------------------------------------------------------------------------
    // Private Interface

    static constexpr int totalNumberVirtualQueues()
    {
      return ((ConcreteOuroboros::QueueType::virtualized) ? ConcreteOuroboros::NumberQueues_ : 0) 
        + Next::totalNumberVirtualQueues();
    }

    static constexpr int totalNumberQueues()
    {
      return ConcreteOuroboros::NumberQueues_ + Next::totalNumberVirtualQueues();
    }

    size_t totalMemoryManagerSize()
    {
      return memory_manager.memoryManagerSize() + next_memory_manager.totalMemoryManagerSize();
    }

    void initMemoryManagers()
    {
      init(&memory);
    }

    void init(Memory* memory)
    {
      memory_manager.initializeNew(&(memory->d_data_end));
      next_memory_manager.init(memory);
    }

    __dpct_inline__ void setMemoryPointer()
    {
      setMemory(&memory);
    }

    __dpct_inline__ void setMemory(Memory *memory)
    {
      memory_manager.d_memory = memory->d_memory;
      memory_manager.d_data = memory->d_data;
      memory_manager.next_free_chunk = &(memory->next_free_chunk);
      memory_manager.maxChunks = memory->maxChunks;
      memory_manager.chunk_locator = &(memory->chunk_locator);
      next_memory_manager.setMemory(memory);
    }

    __dpct_inline__ void initQueues(const Desc&,IndexQueue *d_base_chunk_reuse);

    void printFreeResources();

    __dpct_inline__ void d_printResources();

    __dpct_inline__ bool validOuroborosPointer(void *ptr)
    {
      if(reinterpret_cast<unsigned long long>(ptr) > reinterpret_cast<unsigned long long>(memory.d_memory) 
         && reinterpret_cast<unsigned long long>(ptr) < (reinterpret_cast<unsigned long long>(memory.d_memory) + memory.allocationSize))
        return true;
      return false;
    }

    static bool cuda_initialized;
  };
  template<class OUROBOROS, class... OUROBOROSES>
  bool Ouroboros<OUROBOROS, OUROBOROSES...>::cuda_initialized = false;

  template <>
  struct Ouroboros<>
  {
    void init(Memory* memory) {}
    size_t totalMemoryManagerSize() {return 0ULL;}

    __dpct_inline__ void *malloc(const Desc&,size_t size)
    {
      return nullptr;
    }

    __dpct_inline__ void freePageRecursive(const Desc& d,unsigned int page_size,
                                           MemoryIndex index)
    {
      if(!FINAL_RELEASE)
        d.out<<"Spilled into empty Ouroboros, this should not happend| Page Size: "<<page_size<<
          " | Chunk Index: "<<index.getChunkIndex()<<
          " | Page Index: "<<index.getPageIndex()<<sycl::endl;
      assert(0);
    }

    __dpct_inline__ void setMemory(Memory *memory) {}
    __dpct_inline__ void initQueues(const Desc&,IndexQueue *d_base_chunk_reuse) {}
    __dpct_inline__ void d_printResources() {}
    void printFreeResources(){}
    static constexpr int totalNumberVirtualQueues(){return 0;}
    static constexpr int totalNumberQueues(){return 0;}
    static constexpr bool checkSizeConstraints() { return true;}
  };

  template <typename MemoryManagerType>
  void updateMemoryManagerDevice(MemoryManagerType& memory_manager);
  template <typename MemoryManagerType>
  void updateMemoryManagerHost(MemoryManagerType& memory_manager);
}
