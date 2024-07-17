# Copyright 2021 Matt Borland
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

include(BoostTestJamfile OPTIONAL RESULT_VARIABLE HAVE_BOOST_TEST)

if(HAVE_BOOST_TEST)

    boost_test_jamfile(FILE cmake_jamfile LINK_LIBRARIES Boost::cuda_math Boost::assert Boost::concept_check Boost::config Boost::core Boost::integer Boost::lexical_cast Boost::multiprecision Boost::predef Boost::random Boost::static_assert Boost::throw_exception )

endif()
