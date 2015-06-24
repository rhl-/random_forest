#include <ayasdi/random_forest.hpp>
#include <ayasdi/math.hpp>
#include <cstdlib>
#include <ctime>

namespace math = ayasdi::math;
namespace ml = ayasdi::ml;
namespace io = math::io;

int main(int argc, char ** argv){
    if( argc != 3){
        std::cerr << "Usage: " << argv[ 0] << " train_data.csv test_data.csv" << std::endl; 
        return 0;
    }
    typedef math::Matrix< double, math::Column_major> Column_major_matrix;
    typedef math::Matrix< double> Row_major_matrix;
    typedef math::Vector_facade< double> Output_vector;
    typedef math::Vector_facade< double> Datapoint;
    typedef math::Matrix_facade< double, math::Column_major> Matrix_facade;
    typedef ml::random_forest< Column_major_matrix, Output_vector> random_forest;
    Column_major_matrix A;
    Row_major_matrix B;
    bool fail = io::read_matrix( std::string( argv[ 1]), A);
    if( fail) { 
        std::cout << std::endl; 
        return fail; 
    }
    fail = io::read_matrix( std::string( argv[ 2]), B);
    if( fail) { 
        std::cout << std::endl; 
        return fail; 
    }
    std::cout << "Matrix A: " << A.m() << " " << A.n() << std::endl;
    random_forest rf;
    Matrix_facade X( A.m(), A.n()-1, &*A.begin( 1));
    Output_vector y( A.m(), &*A.begin( 0));
    std::srand(std::time(0));
    ayasdi::timer t;
    t.start();
    rf.train( A, y);
    t.stop();
    std::cout << "Training Time: " << t.elapsed() << std::endl; 
    t.start();
    for( std::size_t i = 0; i < B.m(); ++i){
        Datapoint z( B.n()-1, &*B.begin( i)++);
        rf.classify( z);
    }
    t.stop();
    std::cout << "Testing Time: " << t.elapsed() << std::endl; 
    return 0;
}
