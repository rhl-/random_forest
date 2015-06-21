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

template< typename Row_index_iterator, typename Output>
bool is_pure_column( Row_index_iterator begin, Row_index_iterator end, Output& output){
    auto r = output[ *begin];
    for( ++begin; begin != end; ++begin){
        if( r != output[ *begin]){
            return false;
        }
    }
    return true;
}

public:
 template< typename Row_index_iterator>
 void build_random_tree( Row_index_iterator begin, Row_index_iterator end, 
                         Dataset& dataset, Output& output, Tree& tree){
    current_tree.insert_root();
    std::vector< std::size_t> columns( column_subset_size);
    if( is_pure_column( begin, end, output)){
        //Not possible to split, decision is made.
        //TODO: generate_leaf_node();
        return;
    } 
    if( std::distance(begin, end) < SMALL_DATA_SIZE){
      //Too expensive to split, create a leaf and give it a majority decision
    }
    //randomly pick a subset of size column_subset_size
    random_sample( counting_iterator( 0), counting_iterator( dataset.n()), 
                   columns.begin(), columns.end());
    std::pair< std::size_t, double> overall_best_split( 0, std::numeric_limits< double>::inf());
    for(auto& column: columns){
        std::pair< std::size_t, double> 
        split_and_entropy = find_best_column_split( dataset.begin( column), dataset.end( column), 
                                                    output.begin(), output.end());
        if( split_and_entropy.second < overall_best_split.second){
            overall_best_split = split_and_entropy;
        }
    }
 }

 void train( Column_major_dataset& dataset, Output& output){
    trees.reserve( number_of_trees_);
    for( int i = 0; i < number_of_trees_; ++i){
        auto& current_tree = insert();
        current_tree.reserve( dataset.m());
        build_random_tree( counting_iterator( 0), counting_iterator( dataset.m()), dataset, output, current_tree); 
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
 std::size_t number_of_trees;
}; //end class random_forest

} //end namespace ml
} //end namespace ayasdi

