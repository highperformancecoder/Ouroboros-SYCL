#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once
#include "Definitions.h"

// #######################################################################################
// Unique pointer for device memory
template <typename DataType>
struct CudaUniquePtr
{
	// #######################################################################################
	// Data members
	CudaUniquePtr() : size{ 0 }, data{ nullptr }{}

	CudaUniquePtr(size_t new_size) : size{ new_size }
	{
                HANDLE_ERROR(DPCT_CHECK_ERROR(
                    data = (DataType *)sycl::malloc_device(
                        size * sizeof(DataType), dpct::get_in_order_queue())));
        }

	~CudaUniquePtr()
	{
		if (data)
                        HANDLE_ERROR(DPCT_CHECK_ERROR(
                            dpct::dpct_free(data, dpct::get_in_order_queue())));
        }

	// Disallow copy of any kind, only allow move
	CudaUniquePtr(const CudaUniquePtr&) = delete;
	CudaUniquePtr& operator=(const CudaUniquePtr&) = delete;
	CudaUniquePtr(CudaUniquePtr&& other) noexcept : data{ std::exchange(other.data, nullptr) }, size{ std::exchange(other.size, 0) }{}
	CudaUniquePtr& operator=(CudaUniquePtr&& other) noexcept
	{
		std::swap(data, other.data);
		std::swap(size, other.size);
		return *this;
	}

	// #######################################################################################
	// 
	void allocate(size_t new_size)
        {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
                if (data && size != new_size)
		{
                        HANDLE_ERROR(DPCT_CHECK_ERROR(dpct::dpct_free(data, q_ct1)));
                        data = nullptr;
		}

		size = new_size;
		if (data == nullptr)
                        HANDLE_ERROR(DPCT_CHECK_ERROR(
                            data = (DataType *)sycl::malloc_device(
                                size * sizeof(DataType), q_ct1)));
        }

	// #######################################################################################
	// 
	void resize(size_t new_size)
        {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
                void* tmp{nullptr};
                HANDLE_ERROR(
                    DPCT_CHECK_ERROR(tmp = (void *)sycl::malloc_device(
                                         new_size * sizeof(DataType), q_ct1)));
                HANDLE_ERROR(DPCT_CHECK_ERROR(q_ct1.memcpy(
                    tmp, data, sizeof(DataType) * std::min(size, new_size))));
                HANDLE_ERROR(DPCT_CHECK_ERROR(dpct::dpct_free(data, q_ct1)));
                size = new_size;
		data = reinterpret_cast<DataType*>(tmp);
	}

	// #######################################################################################
	// 
	void copyToDevice(DataType* host_data, size_t copy_size, unsigned int offset = 0)
	{
                HANDLE_ERROR(
                    DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                         .memcpy(data + offset, host_data,
                                                 sizeof(DataType) * copy_size)
                                         .wait()));
        }

	// #######################################################################################
	// 
	void copyFromDevice(DataType* host_data, size_t copy_size, unsigned int offset = 0)
	{
                HANDLE_ERROR(
                    DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                         .memcpy(host_data, data + offset,
                                                 sizeof(DataType) * copy_size)
                                         .wait()));
        }

	// #######################################################################################
	// 
	void memSet(DataType value, size_t memset_size)
	{
                HANDLE_ERROR(DPCT_CHECK_ERROR(
                    dpct::get_in_order_queue()
                        .memset(data, value, sizeof(DataType) * memset_size)
                        .wait()));
        }

	// #######################################################################################
	// 
	explicit operator DataType*() { return data; }

	// #######################################################################################
	// 
	DataType* get() { return data; }

	// #######################################################################################
	// 
	void release()
	{
		if (data)
		{
                        HANDLE_ERROR(DPCT_CHECK_ERROR(
                            dpct::dpct_free(data, dpct::get_in_order_queue())));
                        data = nullptr;
			size = 0;
		}

	}

	// #######################################################################################
	// Data members
	DataType* data{ nullptr };
	size_t size{ 0 };
};