#pragma once
#include <tuple>
#include <vector>

namespace ayasdi{
namespace ml {

//Forward declarations
template< typename Label_type>
class decision_tree; //necessary only for friend declaration below
template< typename Label_type>
class dtree_node{
public:
 dtree_node(){}
 dtree_node( std::size_t split, double split_value, int left_child_index=0, int right_child_index=0):
 split_( split), split_value_( split_value), left_child_index_( left_child_index), right_child_index_( right_child_index) {}

 bool operator!=( const dtree_node& b) const{ return !(this->operator==( b)); }
 bool operator==( const dtree_node& b) const{ 
     if( this == &b) { return true; }
     return (split_value_ == b.split_value_) && (split_ == b.split_) && 
         (left_child_index_ == b.left_child_index_) && (right_child_index_ == b.right_child_index_);
 }

 bool     is_leaf() const { return (left_child_index_ == 0 && 
                                    right_child_index_ == 0); }
 bool is_not_leaf() const { return !is_leaf(); }
 
 //TODO: Figure out the correct behavior for this
 Label_type class_label() const { return (Label_type)split_value_; }

 int  left_child_index() const { return  left_child_index_; }
 int right_child_index() const { return right_child_index_; }
 //TODO: Define invariants for how these values are set for leaf nodes.
 std::size_t split_=0; 
 double split_value_=0;
private:
 int left_child_index_=0;
 int right_child_index_=0;   
 friend decision_tree< Label_type>;
}; //end struct dtree_node

template< typename Label_type>
class decision_tree {
public:
 typedef dtree_node< Label_type> node;

 //Default Constructor
 decision_tree( int reserve_max_size=500){ tree_nodes.reserve( reserve_max_size); }

 //Copy Constructor
 decision_tree( decision_tree& t): tree_nodes( t.tree_nodes){}

 //Move Constructor
 decision_tree( decision_tree&& t): tree_nodes( std::move( t.tree_nodes)){}
 
 //Assignment operator
 decision_tree& operator=( const decision_tree& f) { tree_nodes = f.tree_nodes; }

 //Move Assignment operator
 decision_tree& operator=( decision_tree&& f) { tree_nodes = std::move( f.tree_nodes); }

 //Access a node
 node& operator[]( std::size_t i){ return tree_nodes[ i]; }

/**
 * Inserts left child
 */
 node& insert_left_child(node& parent){
     parent.left_child_index_ = tree_nodes.size();
     return insert();
 }

 /**
 * Inserts right child
 */
 node& insert_right_child(node& parent){
     parent.right_child_index_ = tree_nodes.size();
     return insert();
 }

/** 
 * inserts the left and right node into the tree
 */
 std::tuple<node&,node&>
 insert_children( node& parent){
     auto& left_child = insert_left_child( parent);
     auto& right_child = insert_right_child( parent);		
     return std::forward_as_tuple(left_child, right_child);
 }

 /**
 * Input: a datapoint which provides a double operator[]()
 * This function walks the decision tree and returns the 
 * class_label() supported by this tree. 
 */
 template< typename Datapoint>
 Label_type vote( Datapoint& p){
    node& current_node = root();
    while( current_node.is_not_leaf()){
        if( p[ current_node.split_] < current_node.split_value_){
            current_node = tree_nodes[ current_node.left_child_index()];
        }else{ 
            current_node = tree_nodes[ current_node.right_child_index()];
        }
    }
    return current_node.class_label();
 }

 /**
 * Returns the root of the tree.
 * In debug mode checks that the tree is nonempty first.
 */
 node& root() { 
     #ifdef NDEBUG
     if( tree_nodes.empty()){ std::cerr << "Bug in use of decision_tree." << std::endl; }
     #endif
     return tree_nodes[ 0]; 
 }
 
 /**
 * inserts the root node into the tree if empty.
 * otherwise equivalent to a call to root()
 */
 node& insert_root(){
     if( tree_nodes.size() > 0){ return root(); }
     return insert();
 }

 /**
 * Reserve space for the tree.
 */
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
