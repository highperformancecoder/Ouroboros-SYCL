#pragma once
#include "Parameters.h"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "PageQueueVA.dp.hpp"
#include "device/BulkSemaphore_impl.dp.hpp"
#include "device/Chunk.dp.hpp"
#include "device/MemoryIndex.dp.hpp"
#include "device/queues/QueueChunk_impl.dp.hpp"

namespace Ouro
{
  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename MemoryManagerType>
  __dpct_inline__ void
  PageQueueVA<CHUNK_TYPE>::init(const Desc& d,MemoryManagerType *memory_manager)
  {
    for (int i = d.item.get_global_linear_id();
         i < size_;
         i += d.item.get_global_range().size())
      {
        queue_[i] = DeletionMarker<index_t>::val;
      }

  if (d.item.get_global_linear_id() == 0)
      {
        // Allocate 1 chunk per queue in the beginning
        index_t chunk_index{0};
        memory_manager->template allocateChunk<true>(chunk_index);
        auto chunk = QueueChunkType::initializeChunk(memory_manager->d_data, chunk_index, 0);
        queue_[0] = chunk_index;
      }
  }

  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename MemoryManagerType>
  __dpct_inline__ bool
  PageQueueVA<CHUNK_TYPE>::enqueueChunk(const Desc& d, MemoryManagerType *memory_manager,
                                        index_t chunk_index,
                                        index_t pages_per_chunk)
  {
    if (semaphore.signalExpected(pages_per_chunk) < num_spots_)
      {
        unsigned int virtual_pos = atomicAdd(&back_, pages_per_chunk);
        unsigned int chunk_id = computeChunkID(virtual_pos);
        auto position = Ouro::modPower2<QueueChunkType::num_spots_>(virtual_pos);

        // Do we have to allocate a new pre-emptive chunk
        if((position + pages_per_chunk > QueueChunkType::num_spots_) || (position == 0))
          {
            // We have to pre-allocate a new chunk for the queue here
            unsigned int new_queue_index{0U};
            memory_manager->template allocateChunk<true>(new_queue_index);
            QueueChunkType::initializeChunk(memory_manager->d_data, new_queue_index, virtual_pos + QueueChunkType::num_spots_ + ((position != 0) ? (QueueChunkType::num_spots_ - position) : 0));

            // Please do not re-order here
            /*
              DPCT1078:1: Consider replacing memory_order::acq_rel
              with memory_order::seq_cst for correctness if strong
              memory order restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::seq_cst,
                               sycl::memory_scope::work_group);

            atomicExch(&queue_[Ouro::modPower2<size_>(position != 0 ? (chunk_id + 2) : (chunk_id + 1))], new_queue_index);
          }

        /*
          DPCT1078:0: Consider replacing memory_order::acq_rel with
          memory_order::seq_cst for correctness if strong memory order
          restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::work_group);

        QueueChunkType* queue_chunk{ accessQueueElement(d,memory_manager, chunk_id, virtual_pos) };
        for(auto i = 0; i < pages_per_chunk; ++i)
          {
            if(position == 0)
              {
                // Get new chunk ID
                chunk_id = computeChunkID(virtual_pos);
                queue_chunk = accessQueueElement(d,memory_manager, chunk_id, virtual_pos);
              }

            index_t index = MemoryIndex::createIndex(chunk_index, i);
            if(QueueChunkType::checkChunkEmptyEnqueue(queue_chunk->enqueue(d,position, index)))
              {
                // We can remove this chunk
                index_t reusable_chunk_id = atomicExch(queue_ + chunk_id, DeletionMarker<index_t>::val);
                if(!FINAL_RELEASE && printDebug)
                  d.out<<"We can reuse this chunk: "<<reusable_chunk_id<<
                    " at position: "<<chunk_id<<
                    " with virtual start: "<<queue_chunk->virtual_start_<<sycl::endl;
                memory_manager->template enqueueChunkForReuse<true>(reusable_chunk_id);
              }

            // Compute new index
            ++virtual_pos;
            position = virtual_pos % QueueChunkType::num_spots_;
          }
        return true;
      }

    if (!FINAL_RELEASE)
      d.out<<"We died in EnqueueChunk\n";

    // no space to enqueue -> fail
    return false;
  }

  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename MemoryManagerType>
  __dpct_inline__ bool PageQueueVA<CHUNK_TYPE>::enqueueInitialChunk(
                                                                    MemoryManagerType *memory_manager, index_t chunk_index, int available_pages,
                                                                    index_t pages_per_chunk)
  {
    const auto start_page_index = pages_per_chunk - available_pages;

    for(auto i = start_page_index; i < pages_per_chunk; ++i)
      {
        // Get virtual position and move back along
        unsigned int virtual_pos = back_++;
        unsigned int chunk_id = computeChunkID(virtual_pos);

        if((virtual_pos % QueueChunkType::num_spots_) == 0)
          {
            // We have to allocate a new chunk for the queue here
            unsigned int new_queue_index{0U};
            memory_manager->allocateChunk<true>(new_queue_index);
            auto testchunk = QueueChunkType::initializeChunk(memory_manager->d_data, new_queue_index, virtual_pos + QueueChunkType::num_spots_);
			
            // Please do not re-order here
            /*
              DPCT1078:2: Consider replacing memory_order::acq_rel
              with memory_order::seq_cst for correctness if strong
              memory order restrictions are needed.
            */
            sycl::atomic_fence(sycl::memory_order::acq_rel,
                               sycl::memory_scope::work_group);

            queue_[(chunk_id + 1) % size_] = new_queue_index;
          }

        auto chunk = QueueChunkType::getAccess(memory_manager->d_data, queue_[chunk_id]);
        chunk->enqueueInitial(virtual_pos % QueueChunkType::num_spots_, MemoryIndex::createIndex(chunk_index, i));
      }

    semaphore.signal(available_pages);
    return true;
  }

  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename MemoryManagerType>
  __dpct_inline__ void *
  PageQueueVA<CHUNK_TYPE>::allocPage(const Desc& d,MemoryManagerType *memory_manager)
  {
    using ChunkType = typename MemoryManagerType::ChunkType;

    MemoryIndex index;
    uint32_t chunk_index;
    auto pages_per_chunk = MemoryManagerType::QI::getPagesPerChunkFromQueueIndex(queue_index_);

    semaphore.wait(d,1, pages_per_chunk, [&]()
    {
      if (!memory_manager->template allocateChunk<false>(chunk_index))
        {
          if(!FINAL_RELEASE)
            d.out<<"TODO: Could not allocate chunk!!!\n";
        }

      ChunkType::initializeChunk(memory_manager->d_data, chunk_index, pages_per_chunk);
      //__threadfence();
      sycl::atomic_fence(sycl::memory_order::seq_cst,sycl::memory_scope::device);
      enqueueChunk(d,memory_manager, chunk_index, pages_per_chunk);
    });

    // unsigned int virtual_pos = atomicAdd(&front_, 1);
    unsigned int virtual_pos = Ouro::atomicAggInc(&front_);
    unsigned int chunk_id = computeChunkID(virtual_pos);

    // Get index from queue
    auto chunk = accessQueueElement(d,memory_manager, chunk_id, virtual_pos);
    auto chunk_empty = chunk->dequeue(d,(virtual_pos % QueueChunkType::num_spots_), index.index, memory_manager, nullptr);

    if(chunk_empty)
      {
        // We can remove this chunk
        //index_t reusable_chunk_id = atomicExch(queue_ + chunk_id, DeletionMarker<index_t>::val);
        index_t reusable_chunk_id = Ouro::Atomic<unsigned>(queue_[chunk_id]).exchange(DeletionMarker<index_t>::val);
        if(!FINAL_RELEASE && printDebug)
          d.out<<"We can reuse this chunk: "<<reusable_chunk_id<<
            " at position: "<<chunk_id<<
            " with virtual start: "<<chunk->virtual_start_<<" | AllocPage-Reuse\n";
        memory_manager->template enqueueChunkForReuse<true>(reusable_chunk_id);
      }

    chunk_index = index.getChunkIndex();
    return ChunkType::getPage(memory_manager->d_data, chunk_index, index.getPageIndex(), page_size_);
  }

  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename MemoryManagerType>
  __dpct_inline__ void
  PageQueueVA<CHUNK_TYPE>::freePage(const Desc& d,MemoryManagerType *memory_manager,
                                    MemoryIndex index)
  {
    if (semaphore.signal(1) >= num_spots_)
      {
        if (!FINAL_RELEASE)
          d.out << "We died in FreePage\n";
        assert(0); // no space to enqueue -> fail
      }

    // unsigned int chunk_index, page_index;
    // index.getIndex(chunk_index, page_index);
    enqueue(d,memory_manager, index.index);
  }

  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename MemoryManagerType>
  __dpct_inline__ void
  PageQueueVA<CHUNK_TYPE>::enqueue(const Desc& d,MemoryManagerType *memory_manager,
                                   index_t index)
  {
    // const unsigned int virtual_pos = atomicAdd(&back_, 1);
    const unsigned int virtual_pos = Ouro::atomicAggInc(&back_);
    auto chunk_id = computeChunkID(virtual_pos);
    const auto position = (virtual_pos % QueueChunkType::num_spots_);

    if (position == 0)
      {
        unsigned int chunk_index{ 0 };
        // We pre-emptively allocate the next chunk already
        memory_manager->template allocateChunk<true>(chunk_index);
        QueueChunkType::initializeChunk(memory_manager->d_data, chunk_index, virtual_pos + QueueChunkType::num_spots_);

        /*
          DPCT1078:3: Consider replacing memory_order::acq_rel with
          memory_order::seq_cst for correctness if strong memory order
          restrictions are needed.
        */
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

        atomicExch(&queue_[(chunk_id + 1) % size_], chunk_index);
      }

    auto chunk = accessQueueElement(d,memory_manager, chunk_id, virtual_pos);
    if(QueueChunkType::checkChunkEmptyEnqueue(chunk->enqueue(d, position, index)))
      {
        // We can remove this chunk
        index_t reusable_chunk_id = atomicExch(queue_ + chunk_id, DeletionMarker<index_t>::val);
        Ouro::sleep();
        if(!FINAL_RELEASE && printDebug)
          d.out<<"We can reuse this chunk: "<<reusable_chunk_id<<" at position: "<<chunk_id<<" with virtual start: "<<chunk->virtual_start_<<" | ENQUEUE-Reuse\n";
        memory_manager->template enqueueChunkForReuse<true>(reusable_chunk_id);
      }
  }

  // ##############################################################################################################################################
  //
  template <typename CHUNK_TYPE>
  template <typename MemoryManagerType>
  __dpct_inline__ QueueChunk<typename CHUNK_TYPE::Base> *
  PageQueueVA<CHUNK_TYPE>::accessQueueElement(const Desc& d,MemoryManagerType *memory_manager,
                                              index_t chunk_id,
                                              index_t v_position)
  {
    index_t queue_chunk_index{0};
    // We may have to wait until the first thread on this chunk has initialized it!
    unsigned int counter{0U};
    while((queue_chunk_index = Ouro::ldg_cg(&queue_[chunk_id])) == DeletionMarker<index_t>::val) 
      {
        Ouro::sleep(counter++);
      }

    auto queue_chunk = QueueChunkType::getAccess(memory_manager->d_data, queue_chunk_index);
    if(!queue_chunk->checkVirtualStart(v_position))
      {
        if (!FINAL_RELEASE)
          d.out<<"Virtualized does not match for chunk: "<<queue_chunk_index<<
            " at position: "<<chunk_id<<
            " with virtual start: "<<queue_chunk->virtual_start_<<
            "  ||| v_pos: "<<v_position<<sycl::endl;
        return nullptr;
      }

    return queue_chunk;
  }
}
