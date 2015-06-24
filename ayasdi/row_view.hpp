#pragma once

#include <iterator>
#include <boost/iterator/iterator_facade.hpp>
namespace ayasdi{

template< typename RandomAccessIterator>
class row_view{
    public:
    row_view( RandomAccessIterator i, int stride): it( i), n( stride) {}
    typename std::iterator_traits< RandomAccessIterator>::reference
    operator[]( std::size_t i){ return *(it + n*i); }
    private:
    RandomAccessIterator it;
    int n=1;
}; //end class row_view

} //end namespace ayasdi
