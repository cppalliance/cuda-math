//  Copyright (c) 2024 Matt Borland
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_VECTOR_HPP
#define BOOST_MATH_TOOLS_VECTOR_HPP

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/cstdint.hpp>
#include <boost/math/tools/type_traits.hpp>

#ifndef BOOST_MATH_ENABLE_CUDA

#include <memory>

namespace boost {
namespace math {

using std::shared_ptr;
using std::make_shared;

} // namespace math
} // namespace boost

#else // CUDA shared pointer

#include <cuda_runtime.h>
#include <memory>
#include <type_traits>

#define BOOST_MATH_CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace boost {
namespace math {

// Forward declaration of the control block
template <typename T>
struct ControlBlock;

// CUDA-compatible shared_ptr
template <typename T>
class shared_ptr {
private:
    T* ptr;
    ControlBlock<T>* control_block;

public:
    __host__ __device__
    shared_ptr() : ptr(nullptr), control_block(nullptr) {}

    __host__ __device__
    shared_ptr(T* p, ControlBlock<T>* cb) : ptr(p), control_block(cb) 
    {
        if (control_block) 
        {
            #ifdef __CUDA_ARCH__
            atomicAdd(&control_block->ref_count, 1);
            #else
            ++control_block->ref_count;
            #endif
        }
    }

    __host__ __device__
    ~shared_ptr() 
    {
        if (control_block) 
        {
            #ifdef __CUDA_ARCH__
            if (atomicSub(&control_block->ref_count, 1) == 1)
            {
                ptr->~T();
                free(control_block);
            }
            #else
            if (--control_block->ref_count == 0)
            {
                ptr->~T();
                cudaFree(control_block);
            }
            #endif
        }
    }

    __host__ __device__
    T* get() const { return ptr; }

    __host__ __device__
    T& operator*() const { return *ptr; }

    __host__ __device__
    T* operator->() const { return ptr; }
};

// Control block for reference counting
template <typename T>
struct ControlBlock 
{
    int ref_count;
    using storage = std::aligned_storage<sizeof(T), alignof(T)>::type;

    __host__ __device__
    ControlBlock() : ref_count(1) {}
};

// CUDA-compatible make_shared
template <typename T, typename... Args>
__host__ shared_ptr<T> make_shared(Args&&... args)
{
    ControlBlock<T>* cb;
    BOOST_MATH_CUDA_CHECK(cudaMallocManaged(&cb, sizeof(ControlBlock<T>)));

    new (cb) ControlBlock<T>();
    T* obj_ptr = reinterpret_cast<T*>(&cb->storage);
    new (obj_ptr) T(std::forward<Args>(args)...);

    return shared_ptr<T>(obj_ptr, cb);
}

} // Namespace math
} // Namespace boost

#endif // CUDA vector

#endif // BOOST_MATH_TOOLS_VECTOR_HPP
