//  Copyright (c) 2024 Matt Borland
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Regular use of <vector> is not supported by CUDA
//  And thrust::device_vector does not work on NVRTC
//  Roll our own implmentation of vector using CUDA 

#ifndef BOOST_MATH_TOOLS_VECTOR_HPP
#define BOOST_MATH_TOOLS_VECTOR_HPP

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/cstdint.hpp>
#include <boost/math/tools/type_traits.hpp>

#ifndef BOOST_MATH_ENABLE_CUDA

#include <vector>

namespace boost {
namespace math {

using std::vector;

} // namespace math
} // namespace boost

#else // CUDA capable vector

namespace boost {
namespace math {

template <typename T>
class vector {
private:
    T* data;
    boost::math::size_t capacity_;
    boost::math::size_t current_;

public:
    using size_type = boost::math::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    BOOST_MATH_GPU_ENABLED vector()
    {
        // cudaMalloc takes void** instead of void* like malloc
        const auto res = cudaMalloc(&data, sizeof(T));
        
        if (res == cudaError::cudaSuccess)
        {
            capacity_ = 1;
        }
        else
        {
            capacity_ = 0;
        }

        current_ = 0;
    }

    BOOST_MATH_GPU_ENABLED ~vector()
    {
        if (data != nullptr)
        {
            clear();
            cudaFree(data);
            data = nullptr;
        }
    }

    // Returns true or false based on if the operation succeeded or not
    BOOST_MATH_GPU_ENABLED bool push_back(T element)
    {
        if (current_ == capacity_)
        {
            T* temp;
            const auto res = cudaMalloc(&temp, 2 * capacity_ * sizeof(T));

            if (res == cudaError::cudaSuccess)
            {
                // We have no more memory so we can't add the element
                // We also can't throw or write to errno to signal ENOMEM
                return false;
            }

            for (boost::math::size_t i = 0; i < capacity_; ++ i)
            {
                temp[i] = data[i];
            }

            cudaFree(data);
            capacity_ *= 2;
            data = temp;
        }

        data[current_] = element;
        ++current_;

        return true;
    }

    // Returns true or false based on if the operation succedded or not)
    BOOST_MATH_GPU_ENABLED bool resize(boost::math::size_t new_size, T default_elem)
    {
        if (data != nullptr)
        {
            cudaFree(data);
        }

        const auto res = cudaMalloc(&data, new_size * sizeof(T));
        
        if (res == cudaError::cudaSuccess)
        {
            current_ = 0;
            capacity_ = new_size;

            // cudaMemset only works with ints so we need to loop
            for (boost::math::size_t i = 0; i < capacity_; ++i)
            {
                data[i] = default_elem;
            }

            return true;
        }

        return false;
    }

    BOOST_MATH_GPU_ENABLED bool resize(boost::math::size_t new_size)
    {
        return resize(new_size, static_cast<T>(0));
    }

    BOOST_MATH_GPU_ENABLED void clear()
    {
        // Destroy all the elements (if applicable), but don't resize the vector
        BOOST_MATH_IF_CONSTEXPR (!boost::math::is_trivial_v<T>)
        {
            for (boost::math::size_t i = 0; i < current_; ++i) 
            {
                data[i].~T();
            }
        }

        current_ = 0;
    }

    BOOST_MATH_GPU_ENABLED void pop_back()
    {
        if (current_ > 0)
        {
            --current_;
        }
    }

    BOOST_MATH_GPU_ENABLED T& operator[](boost::math::size_t index)
    {
        return data[index];
    }

    BOOST_MATH_GPU_ENABLED boost::math::size_t size() const
    {
        return current_;
    }

    BOOST_MATH_GPU_ENABLED boost::math::size_t capacity() const
    {
        return capacity_;
    }

    BOOST_MATH_GPU_ENABLED bool empty() const 
    {
        return current_ == 0;
    }

    BOOST_MATH_GPU_ENABLED T* begin()
    {
        return data;
    }

    BOOST_MATH_GPU_ENABLED T* begin() const
    {
        return data;
    }

    BOOST_MATH_GPU_ENABLED T* end()
    {
        return data + current_;
    }

    BOOST_MATH_GPU_ENABLED T* end() const
    {
        return data + current_;
    }
};

} // Namespace math
} // Namespace boost

#endif // CUDA vector

#endif // BOOST_MATH_TOOLS_VECTOR_HPP
