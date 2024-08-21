
///////////////////////////////////////////////////////////////////////////////
//  Copyright 2013 Nikhar Agrawal
//  Copyright 2013 Christopher Kormanyos
//  Copyright 2013 John Maddock
//  Copyright 2013 Paul Bristow
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_BERNOULLI_B2N_2013_05_30_HPP_
#define _BOOST_BERNOULLI_B2N_2013_05_30_HPP_

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/cstdint.hpp>
#include <boost/math/tools/type_traits.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/detail/unchecked_bernoulli.hpp>
#include <boost/math/special_functions/detail/bernoulli_details.hpp>
#include <boost/math/policies/error_handling.hpp>

namespace boost { namespace math {

namespace detail {

template <class T, class OutputIterator, class Policy, int N>
BOOST_MATH_GPU_ENABLED OutputIterator bernoulli_number_imp(OutputIterator out, boost::math::size_t start, boost::math::size_t n, const Policy& pol, const boost::math::integral_constant<int, N>& tag)
{
   for(boost::math::size_t i = start; (i <= max_bernoulli_b2n<T>::value) && (i < start + n); ++i)
   {
      *out = unchecked_bernoulli_imp<T>(i, tag);
      ++out;
   }

   for(boost::math::size_t i = BOOST_MATH_GPU_SAFE_MAX(static_cast<boost::math::size_t>(max_bernoulli_b2n<T>::value + 1), start); i < start + n; ++i)
   {
      // We must overflow:
      *out = (i & 1 ? 1 : -1) * policies::raise_overflow_error<T>("boost::math::bernoulli_b2n<%1%>(n)", nullptr, T(i), pol);
      ++out;
   }
   return out;
}

template <class T, class OutputIterator, class Policy>
BOOST_MATH_GPU_ENABLED OutputIterator bernoulli_number_imp(OutputIterator out, boost::math::size_t start, boost::math::size_t n, const Policy& pol, const boost::math::integral_constant<int, 0>& tag)
{
   for(boost::math::size_t i = start; (i <= max_bernoulli_b2n<T>::value) && (i < start + n); ++i)
   {
      *out = unchecked_bernoulli_imp<T>(i, tag());
      ++out;
   }
   //
   // Short circuit return so we don't grab the mutex below unless we have to:
   //
   if(start + n <= max_bernoulli_b2n<T>::value)
   {
      return out;
   }

   return get_bernoulli_numbers_cache<T, Policy>().copy_bernoulli_numbers(out, start, n, pol);
}

} // namespace detail

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline T bernoulli_b2n(const int i, const Policy &pol)
{
   using tag_type = boost::math::integral_constant<int, detail::bernoulli_imp_variant<T>::value>;
   if(i < 0)
   {
      return policies::raise_domain_error<T>("boost::math::bernoulli_b2n<%1%>", "Index should be >= 0 but got %1%", T(i), pol);
   }

   T result {};
   boost::math::detail::bernoulli_number_imp<T>(&result, static_cast<boost::math::size_t>(i), 1u, pol, tag_type());
   return result;
}

template <class T>
BOOST_MATH_GPU_ENABLED inline T bernoulli_b2n(const int i)
{
   return boost::math::bernoulli_b2n<T>(i, policies::policy<>());
}

template <class T, class OutputIterator, class Policy>
BOOST_MATH_GPU_ENABLED inline OutputIterator bernoulli_b2n(const int start_index,
                                    const unsigned number_of_bernoullis_b2n,
                                    OutputIterator out_it,
                                    const Policy& pol)
{
   using tag_type = boost::math::integral_constant<int, detail::bernoulli_imp_variant<T>::value>;
   if(start_index < 0)
   {
      *out_it = policies::raise_domain_error<T>("boost::math::bernoulli_b2n<%1%>", "Index should be >= 0 but got %1%", T(start_index), pol);
      return ++out_it; // LCOV_EXCL_LINE we don't reach here, previous line throws.
   }

   return boost::math::detail::bernoulli_number_imp<T>(out_it, start_index, number_of_bernoullis_b2n, pol, tag_type());
}

template <class T, class OutputIterator>
BOOST_MATH_GPU_ENABLED inline OutputIterator bernoulli_b2n(const int start_index,
                                    const unsigned number_of_bernoullis_b2n,
                                    OutputIterator out_it)
{
   return boost::math::bernoulli_b2n<T, OutputIterator>(start_index, number_of_bernoullis_b2n, out_it, policies::policy<>());
}

template <class T, class Policy>
BOOST_MATH_GPU_ENABLED inline T tangent_t2n(const int i, const Policy &pol)
{
   if(i < 0)
   {
      return policies::raise_domain_error<T>("boost::math::tangent_t2n<%1%>", "Index should be >= 0 but got %1%", T(i), pol);
   }

   T result {};
   boost::math::detail::get_bernoulli_numbers_cache<T, Policy>().copy_tangent_numbers(&result, i, 1, pol);
   return result;
}

template <class T>
BOOST_MATH_GPU_ENABLED inline T tangent_t2n(const int i)
{
   return boost::math::tangent_t2n<T>(i, policies::policy<>());
}

template <class T, class OutputIterator, class Policy>
BOOST_MATH_GPU_ENABLED inline OutputIterator tangent_t2n(const int start_index,
                                    const unsigned number_of_tangent_t2n,
                                    OutputIterator out_it,
                                    const Policy& pol)
{
   if(start_index < 0)
   {
      *out_it = policies::raise_domain_error<T>("boost::math::tangent_t2n<%1%>", "Index should be >= 0 but got %1%", T(start_index), pol);
      return ++out_it; // LCOV_EXCL_LINE we don't reach here, previous line throws.
   }

   return boost::math::detail::get_bernoulli_numbers_cache<T, Policy>().copy_tangent_numbers(out_it, start_index, number_of_tangent_t2n, pol);
}

template <class T, class OutputIterator>
BOOST_MATH_GPU_ENABLED inline OutputIterator tangent_t2n(const int start_index,
                                    const unsigned number_of_tangent_t2n,
                                    OutputIterator out_it)
{
   return boost::math::tangent_t2n<T, OutputIterator>(start_index, number_of_tangent_t2n, out_it, policies::policy<>());
}

} } // namespace boost::math

#endif // _BOOST_BERNOULLI_B2N_2013_05_30_HPP_
