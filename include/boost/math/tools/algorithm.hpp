//  Copyright (c) 2024 Matt Borland
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_ALGORITHM_HPP
#define BOOST_MATH_TOOLS_ALGORITHM_HPP

#include <boost/math/tools/config.hpp>

#ifndef BOOST_MATH_ENABLE_CUDA

#include <algorithm>

namespace boost {
namespace math {

using std::fill;
using std::fill_n;

} // namespace math
} // namespace boost

#else

namespace boost {
namespace math {

template <typename ForwardIt, typename T>
BOOST_MATH_GPU_ENABLED void fill(ForwardIt first, ForwardIt last, const T& value)
{
    while (first != last)
    {
        *first = value;
        ++first;
    }
}

template <typename OutputIt, typename Size, typename T>
BOOST_MATH_GPU_ENABLED OutputIt fill_n(OutputIt first, Size count, const T& value)
{
    for (Size i = 0; i < count; i++)
    {
        *first++ = value;
    }

    return first;
}

} // Namespace math
} // Namespace boost

#endif // CUDA algos

#endif // BOOST_MATH_TOOLS_ALGORITHM_HPP
