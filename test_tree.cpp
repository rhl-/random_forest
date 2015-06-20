#include "tree.hpp"
#include <iostream>

namespace ml = ayasdi::ml;

int main(int argc, char** argv){
	ml::decision_tree t;
	auto r = t.insert_root();
 	auto g = t.root();

	std::cout << (g == r) << std::endl;
	auto children = t.insert_children( g);

	auto& left_child = std::get<0>( children);
	auto& right_child = std::get<1>( children);

	return 0;
}
