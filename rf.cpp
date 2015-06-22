#include "random_forest.hpp"
#include <ayasdi/math.hpp>

namespace math = ayasdi::math;
namespace ml = ayasdi::ml;
namespace io = math::io;

int main(int argc, char ** argv)
{
    
    typedef math::Matrix< double, math::Column_major> Column_major_matrix;
    typedef math::Vector_facade< double> Output_vector;
    typedef math::Matrix_facade< double, math::Column_major> Matrix_facade;
    typedef ml::random_forest< Column_major_matrix, Output_vector> random_forest;
    Column_major_matrix A;

    bool fail = io::read_matrix( std::string( argv[ 1]), A);
    if( fail) { 
        std::cout << std::endl; 
        return fail; 
    }
    random_forest rf;
    Matrix_facade X( A.m(), A.n()-1, A.data());
    Output_vector y( A.m(), &*A.begin( A.n()-1));
    rf.train( A, y);
    return 0;
}
