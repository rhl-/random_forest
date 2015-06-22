#include "random_forest.hpp"
#include <ayasdi/math.hpp>

namespace math = ayasdi::math;
namespace io = math::io;

int main(int argc, char ** argv)
{
    
    typedef math::Matrix< double, math::Column_major> Column_major_matrix;
    Column_major_matrix A;

    bool fail = io::read_matrix( std::string( argv[ 1]), A);
    if( fail) { 
        std::cout << std::endl; 
        return fail; 
    }

    return 0;
}
