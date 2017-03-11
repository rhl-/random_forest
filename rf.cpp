#include <random_forest/random_forest.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


PYBIND11_PLUGIN(ml) {
    py::module m("ml", "Machine Learning Code");

    using Rf = ml::Random_forest;

    py::class_<Rf> rf(m, "RandomForest");
    .def("__init__",
      [](Rf &instance, std::list<c> arg) {
          new (&instance) s(std::begin(arg), std::end(arg));
     })
    template< typename T>
    using Matrix = py::array<T, py::array::f_style | py::array::forcecast>;
    
    typedef Matrix<double> DoubleMatrix;
    typedef Matrix<float> FloatMatrix;
    
    rf.def("fit", [](Rf& rfi, DoubleMatrix& X, DoubleMatrix& y){
    	rfi.fit(X,y);
	return rfi; 
    });
    rf.def("fit", [](Rf& rfi, FloatMatrix& X, DoubleMatrix& y){
    	rfi.fit(X,y);
    	return rfi; 
    });
    rf.def("fit", [](Rf& rfi, DoubleMatrix& X, FloatMatrix& y){
    	rfi.fit(X,y);
	return rfi; 
    });
    rf.def("fit", [](Rf& rfi, FloatMatrix& X, FloatMatrix& y){
    	rfi.fit(X,y);
	return rfi; 
    });
    rf.def("predict", [](Rf& rfi, DoubleMatrix& x){ return rfi.predict(x); });
    rf.def("predict", [](Rf& rfi, FloatMatrix& x){ return rfi.predict(x); });
    return m.ptr();
}
