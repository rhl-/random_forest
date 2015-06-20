#pragma once
namespace ayasdi{
namespace ml {
template< typename T, int N>
struct tree_node{
   T t;
   std::array<int, N> children;
};

template< typename T, int N=2>
class tree {
	public:
	typedef tree_node< T, N> node;
	tree( int reserve_max_size): { tree_nodes.reserve( reserve_max_size); }
	node& insert(iterator parent) {
		tree_nodes.emplace_back();
	}
	iterator begin() { }
	const_iterator begin() {}

	iterator end() {}
	const_iterator end() {}
	
	private:
	std::vector< node> tree_nodes; 
}

} //ml namespace
} //ayasdi namespace
