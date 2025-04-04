#include <sycl/sycl.hpp>
//#include <dpct/dpct.hpp>
#pragma once

#include "Definitions.h"
#include "device/Chunk.dp.hpp"
#include "device/ChunkIndexChunk.dp.hpp"

namespace Ouro
{
  template <typename ChunkBase>
  struct QueueChunk : public CommonChunk
  {
    using Base = ChunkBase;
    using QueueDataType = index_t;

    enum class DEQUEUE_MODE
      {
        DEQUEUE,
        DELETE
      }; // Is used to use the same dequeue function with two different internal functions

    enum class Mode
      {
        SINGLE,
        V4
      }; // Enqueue vectorized or single element

    // Static members
    static constexpr auto size_{ChunkBase::size_};
    static constexpr unsigned int num_spots_{ size_ / sizeof(QueueDataType) }; // How many indices can one QueueChunk store
    static constexpr unsigned int shift_value {(sizeof(unsigned int) * BYTE_SIZE) / 2}; // Given count, how many bits given for countA
    static constexpr unsigned int lower_mask {(1 << shift_value) - 1}; // Mask to get countA
    static constexpr unsigned int upper_mask {~lower_mask}; // Mask to get countB
    static constexpr int vector_width{4}; // How large is the vector store unit
    static constexpr unsigned int num_spots_vec4{num_spots_/vector_width}; // How many iterations on a QueueChunk given this vector width
    static constexpr unsigned int chunk_empty_count_dequeue {(num_spots_ + (1 << shift_value))}; // After Dequeue, this value signals chunk empty
    const sycl::uint4 deletionmarker_vec4{
      DeletionMarker<QueueDataType>::val,
      DeletionMarker<QueueDataType>::val,
      DeletionMarker<QueueDataType>::val,
      DeletionMarker<QueueDataType>::val};

    // Members
    QueueDataType* queue_{nullptr}; // Queue data
    unsigned long long next_{DeletionMarker<unsigned long long>::val}; // Pointer to the next chunk (only used in linked-list mode)
    unsigned int count_{0U}; // Two 16 bit counters in one (counterB | counterA)
    unsigned int identifier{ QUEUECHUNK_IDENTIFIER }; // Use to differentiate regular chunks and queue chunks (only needed for debug prints)
    unsigned int virtual_start_; // The virtual start index (e.g. 1024, which means that indices 1024 - 2047 are on this chunk if 1024 indices fit on chunk)
    index_t chunk_index_{0};

    // ##########################################################################################################################
    // ##########################################################################################################################
    // Methods
    // ##########################################################################################################################
    // ##########################################################################################################################
    // ##############################################################################################################################################
    // TODO: We can remove the chunk_index parameter
    inline QueueChunk(QueueDataType *queue, index_t chunk_index,
                               unsigned int virtual_start)
      : CommonChunk{0}, queue_{queue}, chunk_index_{chunk_index},
        virtual_start_{virtual_start} {cleanChunk();}

    // TODO: test with the real version later on
    inline void cleanChunk()
    {
      // for(auto i = 0U; i < num_spots_vec4; ++i)
      // {
      // 	reinterpret_cast<uint4*>(queue_)[i] = deletionmarker_vec4;
      // }
      for(auto i = 0U; i < num_spots_; ++i)
        {
          // queue_[i] = DeletionMarker<QueueDataType>::val;
          atomicExch(&queue_[i], DeletionMarker<QueueDataType>::val);
        }
    }

    // ##############################################################################################################################################
    //
    //        inline void enqueueInitial(const Desc&,const unsigned int position,
    //                                            const QueueDataType element);

    // ##############################################################################################################################################
    //
    template <class Desc>
    inline unsigned int enqueue(const Desc&,const unsigned int position,
                                         const QueueDataType element);

    // ##############################################################################################################################################
    //
    template <class Desc>
    inline unsigned int enqueueLinked(const Desc&,const unsigned int position,
                                               const QueueDataType element);

    // ##############################################################################################################################################
    //
    template <class Desc>
    inline unsigned int
    enqueueLinkedv4(const Desc&,const unsigned int position, const index_t chunk_index,
                    const index_t start_index);

    // ##############################################################################################################################################
    //
    template <typename Desc, typename MemoryManagerType>
    inline void
    enqueue(const Desc&,MemoryManagerType *memory_manager, const unsigned int position,
            const QueueDataType element, QueueChunk **queue_next_ptr,
            QueueChunk **queue_front_ptr, QueueChunk **queue_old_ptr,
            unsigned int *old_count);

    // ##############################################################################################################################################
    //
    template <typename Desc,typename MemoryManagerType>
    inline void
    enqueueChunk(const Desc&, MemoryManagerType *memory_manager,
                 unsigned int start_position, const index_t chunk_index,
                 index_t pages_per_chunk, QueueChunk **queue_next_ptr,
                 QueueChunk **queue_front_ptr, QueueChunk **queue_old_ptr,
                 unsigned int *old_count, int start_index = 0);

    // ##############################################################################################################################################
    //
    template <typename Desc,typename MemoryManagerType>
    inline bool dequeue(const Desc&, const unsigned int position,
                                 QueueDataType &element,
                                 MemoryManagerType *memory_manager,
                                 QueueChunk **queue_front_ptr);

    // ##############################################################################################################################################
    //
    template <class Desc>
    inline bool deleteElement(const Desc&,const unsigned int position);

    // ##############################################################################################################################################
    //
    template <typename Desc, DEQUEUE_MODE Mode, typename MemoryManagerType>
    inline void
    dequeue(const Desc&, MemoryManagerType *memory_manager, const unsigned int position,
            QueueDataType &element, QueueChunk **queue_front_ptr,
            QueueChunk **queue_old_ptr, unsigned int *old_count);

    // ##############################################################################################################################################
    //
    inline void access(const unsigned int position,
                                QueueDataType &element)
    { 
      element = queue_[position]; 
    }

    // ##############################################################################################################################################
    //
    inline void accessLinked(const unsigned int position,
                                      QueueDataType &element);
    template <class Desc>
    inline QueueChunk<ChunkBase> *
    accessLinked(const Desc&,const unsigned int position);

    // ##############################################################################################################################################
    //
    inline bool checkVirtualStart(const unsigned int v_position)
    { 
      // The division is necessary since v_position is just one position on that chunk, so the division rounds it down automatically
      return (virtual_start_ / num_spots_) == (v_position / num_spots_);
    }

    // ##############################################################################################################################################
    //
    inline unsigned int extractCounterA(unsigned int counter)
    {
      return counter & lower_mask;
    }

    // ##############################################################################################################################################
    //
    inline unsigned int extractCounterB(unsigned long long counter)
    {
      return counter >> shift_value;
    }

    // ##############################################################################################################################################
    //
    template <class Desc>
    inline void setBackPointer(const Desc&,QueueChunk **queue_next_ptr);

    // ##############################################################################################################################################
    //
    inline unsigned int setFrontPointer(QueueChunk **queue_front_ptr);

    // ##############################################################################################################################################
    //
    template <typename Desc,typename MemoryManagerType>
    inline void setOldPointer(const Desc&,MemoryManagerType *memory_manager,
                                       QueueChunk **queue_old_ptr,
                                       unsigned int *old_count,
                                       unsigned int free_count);

    // ##############################################################################################################################################
    //
    template <class Desc>
    inline QueueChunk *
    locateQueueChunkForPosition(const Desc&,const unsigned int v_position,
                                const char *message = "");

    // ##############################################################################################################################################
    //
    template <typename Desc,typename FUNCTION>
    inline void guaranteeWarpSyncPerChunk(const Desc&,index_t position,
                                                   const char *message,
                                                   FUNCTION f);

    // ##############################################################################################################################################
    //
    inline bool goToNextChunk(index_t local_position, Mode mode)
    {
      return (local_position == ((mode == Mode::SINGLE) ? (num_spots_ - 1) : (num_spots_ - vector_width)));
    }

    // ##############################################################################################################################################
    //
    inline index_t enqueueChunkAdditionFactor(Mode mode)
    {
      return ((mode == Mode::SINGLE) ? 1 : vector_width);
    }

    // ##########################################################################################################################
    // ##########################################################################################################################
    // STATIC Methods
    // ##########################################################################################################################
    // ##########################################################################################################################

    template <unsigned int ADD_VALUE>
    static constexpr unsigned int countAddValueEnqueue(){return (ADD_VALUE << shift_value) + ADD_VALUE;}

    // ##############################################################################################################################################
    //
    template <unsigned int ADD_VALUE = 1>
    static inline bool
    checkChunkEmptyEnqueue(unsigned int enqueue_count)
    {
      // Check if atomic return value (plus what we added on top of it) is equal to counterB = 0 and counterA = num_spots
      return (enqueue_count + countAddValueEnqueue<ADD_VALUE>()) == num_spots_;
    }

    // ##############################################################################################################################################
    //
    static inline bool checkChunkEmptyDequeue(unsigned int dequeue_count)
    {
      // Check if atomic return value (plus what we added on top of it) is equal to counterB = 0 and counterA = num_spots
      return (dequeue_count - (1 << shift_value)) == (num_spots_);
    }

    // ##############################################################################################################################################
    //
    static inline QueueChunk *getAccess(memory_t *memory,
                                                 index_t chunk_index)
    {
      return reinterpret_cast<QueueChunk*>(Base::getMemoryAccess(memory, chunk_index)); 
    }

    // ##############################################################################################################################################
    //
    static inline QueueDataType *getData(memory_t *memory,
                                                  index_t chunk_index)
    {
      return reinterpret_cast<QueueDataType*>(Base::getData(memory, chunk_index));
    }

    // ##############################################################################################################################################
    //
    static inline QueueChunk *
    initializeChunk(memory_t *memory, index_t chunk_index,
                    unsigned int virtual_start)
    {
      static_assert(Ouro::alignment(sizeof(QueueChunk)) <= CHUNK_METADATA_SIZE, "QueueChunk is larger than alignment!");
      return new (reinterpret_cast<char *>(getAccess(memory, chunk_index)))
        QueueChunk(getData(memory, chunk_index), chunk_index,
                   virtual_start);
    }
  };
}
