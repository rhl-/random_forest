#pragma once
#include <tuple>
#include <vector>

namespace ayasdi{
namespace ml {

//Forward declarations
class decision_tree; //necessary only for friend declaration below

struct dtree_node{
 bool operator==( const dtree_node& b) const{ 
     if( this == &b) { return true; }
     return (split_value == b.split_value) && (split == b.split) && 
         (left_child_index == b.left_child_index) && (right_child_index == b.right_child_index);
 }

 int left_child() const { return left_child_index; }
 int right_child() const { return right_child_index; }
 
 std::size_t split; 
 double split_value;

 private:
 int left_child_index;
 int right_child_index;   
 friend decision_tree;
}; //end struct dtree_node

class decision_tree {
 public:
 typedef dtree_node node;
 
 decision_tree( int reserve_max_size=500){ tree_nodes.reserve( reserve_max_size); }

 node& operator[]( std::size_t i){ return tree_nodes[ i]; }

 node& insert_left_child(node& parent){
     parent.left_child_index = tree_nodes.size();
     return insert();
 }
 
 node& insert_right_child(node& parent){
     parent.right_child_index = tree_nodes.size();
     return insert();
 }
 
 std::tuple<node&,node&>
 insert_children( node& parent){
     auto& left_child = insert_left_child( parent);
     auto& right_child = insert_right_child( parent);		
     return std::forward_as_tuple(left_child, right_child);
 }
 
 node& root() { 
     #ifdef NDEBUG
     if( tree_nodes.empty()){ std::cerr << "Bug in use of decision_tree." << std::endl; }
     #endif
     return tree_nodes[ 0]; 
 }
 
 node& insert_root(){
     if( tree_nodes.size() > 0){ return root(); }
     return insert();
 }
 
 void reserve( int size){ tree_nodes.reserve( size); }
 
private:
 node& insert(){
    tree_nodes.emplace_back(); 
    return tree_nodes.back();
 }
 std::vector<node> tree_nodes; 
}; //end class decision_tree

} //ml namespace
} //ayasdi namespace
