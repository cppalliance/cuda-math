//  Copyright John Maddock 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

#define SC_(x) static_cast<T>(BOOST_STRINGIZE(x))

#include "log1p_expm1_test.hpp"

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the functions log1p and expm1 for
// simple multiprecision types which are able to utilize
// our own rational approximations.  This tests those
// approximations even when float/double/long double
// are not using them.
//

void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //

   //
   // Catch all cases come last:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                          // test type(s)
      ".*",                          // test data group
      ".*",                          // test function
      4,                             // Max Peek error
      3);                            // Max mean error

   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", " 
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}


BOOST_AUTO_TEST_CASE( test_main )
{
   expected_results();

   test(boost::multiprecision::cpp_bin_float_double(0), "UDT double");
   test(boost::multiprecision::cpp_bin_float_double_extended(0), "UDT 80-bit long double");
   test(boost::multiprecision::cpp_bin_float_quad(0), "UDT quad-float");
   test(boost::multiprecision::number<boost::multiprecision::cpp_bin_float<36> >(0), "UDT 36 digits");
}
