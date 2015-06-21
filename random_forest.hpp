#pragma once

//Project
#include "tree.hpp"
#include "random_sample.hpp"

//STL
#include <unordered_map>

//Boost
#include <boost/iterator/counting_iterator.hpp>

namespace ayasdi{
namespace ml{

template< typename Dataset, typename Output>
class random_forest{
public:
 typedef typename Output::value_type Label_type;
private:
 typedef decision_tree< Label_type> tree;
 typedef boost::counting_iterator<std::size_t> counting_iterator;

 template< typename Row_index_iterator>
 bool is_pure_column( Row_index_iterator begin, Row_index_iterator end, Output& output){
     auto r = output[ *begin];
     for( ++begin; begin != end; ++begin){
         if( r != output[ *begin]){
             return false;
         }
     }
     return true;
 }
 
 //Set data of leaf node to be a leaf node.
 //TODO: This needs to be carefully handled.
 void generate_leaf_node( typename tree::node& n, Label_type label){ n.split_value_ = label; }
 
 //Set data of node to be an internal decision node of tree
 void set_split( typename tree::node& n, std::size_t& column_index, double& split_threshold_value){
     n.split_ = column_index;
     n.split_value_ = split_threshold_value;
 }
 
 template< typename Row_index_iterator>
 Label_type get_majority_vote( Row_index_iterator begin, Row_index_iterator end, Output& output){
     typedef std::unordered_map< Label_type, std::size_t> Map;
     Map votes;
     for( ; begin != end; ++begin){ votes[ *begin]++; }
     typedef typename Map::value_type pair;
     auto& max_elt= std::max( votes.begin(), votes.end(), 
                              [&](pair& a, pair& b){ return a.second < b.second; });
     return max_elt.first;
 }
public:
 template< typename Row_index_iterator>
 void build_random_tree( Row_index_iterator begin, Row_index_iterator end, 
                         Dataset& dataset, Output& output, tree& t, typename tree::node& n){
    typedef std::vector< std::size_t> Vector;
    
    //Not possible to split, decision is already made.
    //Create a leaf node with this decision
    if( is_pure_column( begin, end, output)){
        Label_type class_label = output[ *begin];
        generate_leaf_node(n, class_label);
        return;
    }

    //TODO: Check if assuming log( number of data elements) is appropriate
    //Data is too small to waste time splitting. We punt.
    //Create a leaf node and give it a majority decision
    if( std::distance(begin, end) < std::log( dataset.m())){
        auto& class_label = get_majority_vote( begin, end, output);
        generate_leaf_node(n, class_label);
        return;
    }

    Vector columns( column_subset_size_);
    //randomly pick a subset of size column_subset_size
    //This is inefficient since we have R.A. to [0,n]
    random_sample( counting_iterator( 0), counting_iterator( dataset.n()), 
                   columns.begin(), columns.end());
    //Output Variables
    double best_entropy=std::numeric_limits< double>::infinity();
    std::size_t column_index_for_split=std::numeric_limits< std::size_t>::infinity();
    double split_threshold_value=best_entropy;
    std::tuple< Vector, Vector> row_indices_for_split;
    //Find the best split within each column, find minimal overall split.
    for(auto& column: columns){
        //TODO: Implement this
        std::pair< std::size_t, double> 
        split_and_entropy = find_best_column_split( dataset.begin( column), dataset.end( column), 
                                                    begin, end, //Obs: We may sort this all we like.
                                                    output.begin(), output.end());
        if( split_and_entropy.second < best_entropy){
            //Record the entropy so far and which column we are in
            best_entropy = split_and_entropy.second;
            column_index_for_split = column;

            //Index into sorted range of split.
            std::size_t split_index = split_and_entropy.first;

            //Get the entry containing the split_threshold_value
            split_threshold_value = dataset.begin( column)+*(begin+split_index);

            //Since we resort at every step we make physical copies of the row indices
            //If the column was gaurunteed sorted then we could skip this!
            //This seems like it should save a lot of memory..!
            row_indices_for_split = std::move( std::make_tuple( Vector( begin, begin+split_index),
                                                                Vector( begin+split_index, end)));
        }
    }
    //Build the split into the tree
    set_split( n, column_index_for_split, split_threshold_value);
    //add children nodes into Decision Tree
    auto& kids = t.insert_children( n);
    //Recursively call.
    auto& left_indices = std::get< 0>(row_indices_for_split);
    auto& right_indices = std::get< 1>(row_indices_for_split);
    build_random_tree( left_indices.begin(), left_indices.end(), dataset, output, t, std::get<0>(kids));
    build_random_tree( right_indices.begin(), right_indices.end(), dataset, output, t, std::get<1>(kids));
 }

 template< typename Column_major_dataset>
 void train( Column_major_dataset& dataset, Output& output){
    trees.reserve( number_of_trees_);
    for( int i = 0; i < number_of_trees_; ++i){
        auto& current_tree = insert();
        current_tree.reserve( dataset.m());
        auto& root = current_tree.insert_root();
        build_random_tree( counting_iterator( 0), counting_iterator( dataset.m()), 
                           dataset, output, current_tree, root); 
    }
 }

 template< typename Datapoint>
 Label_type classify( Datapoint& p) const{
     typedef std::unordered_map< Label_type, std::size_t> Map; 
     Map votes;
     for( auto & tree: trees){ votes[ tree.vote( p)]++; }
     typedef typename Map::value_type pair;
     std::max( votes.begin(), votes.end(), [&](pair& a, pair& b){ return a.second < b.second; });
 }
 
private:
 tree& insert(){
    trees.emplace_back();
    return trees.back();
 }
 std::vector< tree> trees;
 std::size_t number_of_trees_;
 std::size_t column_subset_size_;
}; //end class random_forest

} //end namespace ml
} //end namespace ayasdi

