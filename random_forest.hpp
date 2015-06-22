#pragma once

//Project
#include "tree.hpp"
#include "random_sample.hpp"

//STL
#include <unordered_map>
#include <cmath>

//Boost
#include <boost/iterator/counting_iterator.hpp>

#define MAX_TREE_HEIGHT 30

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
                              [&](const pair& a, const pair& b){ return a.second < b.second; });
     return max_elt.first;
 }

 template< typename Counts>
 double entropy( const Counts& counts, std::size_t& n){
     double denominator = 1.0/n;
     double entropy=0.0;
     for( auto& key_value_pair: counts){
         double p = (key_value_pair.second)*denominator;
         entropy -= p*std::log( p);
     }
     return entropy;
 }

 template< typename Counts>
 double balanced_entropy( Counts& lower_counts, Counts& upper_counts, 
                          std::size_t lower_index, std::size_t upper_index, 
                          std::size_t number_of_rows){
     double lower_entropy = entropy( lower_counts, lower_index);
     double upper_entropy = entropy( upper_counts, upper_index);
     return ((1.0/lower_index)*lower_entropy + (1.0/(upper_index))*upper_entropy)/number_of_rows;
 }

 template< typename Column_iterator, typename Row_index_iterator, typename Output_column_iterator>
 std::pair< std::size_t, double>
 find_best_column_split( Column_iterator col_begin, Column_iterator col_end,
                          //Observation: we may sort the row iterators safely
                          Row_index_iterator row_idx_begin, Row_index_iterator row_idx_end,
                          Output_column_iterator output_begin, Output_column_iterator output_end){
    typedef std::unordered_map< Label_type, std::size_t> Map; 
    //We just sort the row indices into order
    //We can GPU accelerate this for fun with thrust::sort()
    //Also we can try tbb::sort()
    std::sort( row_idx_begin, row_idx_end, 
                 [&](const std::size_t& a, const std::size_t& b){ return *(col_begin+a) < *(col_begin+b);});
    Map lower_counts, upper_counts;
    for( ; row_idx_begin != row_idx_end; ++row_idx_begin) { upper_counts[ *row_idx_begin]++; }
   
    std::size_t number_of_rows=std::distance(row_idx_begin,row_idx_end);
    std::pair< std::size_t, double> best_split(-1, std::numeric_limits< double>::infinity());
    //By assumption at this point number_of_rows > 1
    best_split.second = balanced_entropy( lower_counts, upper_counts, 1, number_of_rows-1, number_of_rows);

    std::size_t category_width=0;
    for( auto split_index = row_idx_begin+1; split_index != row_idx_end;  ++split_index){
        //TODO: Compute split optimization function Entropy, Gini, etc.
        auto class_label = *(output_begin+*split_index);
        lower_counts[class_label]++;
        upper_counts[class_label]--;
        //This logic handles repeated values in the input column
        if( col_begin+split_index == col_begin+(split_index-1)){ 
            category_width++;
            continue; 
        }else{ category_width = 0; } 
        auto lower_index = std::distance(row_idx_begin,split_index)+1;
        auto upper_index = number_of_rows-lower_index;
        auto current_entropy = balanced_entropy( lower_counts, upper_counts, lower_index, upper_index, number_of_rows);
        if( current_entropy < best_split.second){     
           best_split.first  = *split_index;
           best_split.second = current_entropy;
           if( current_entropy == 0.0 ){ return best_split; }
        }
    }
    return best_split;
 }                                                                                                                 
                                                                                                                   
public:
 std::size_t max_tree_height() const                   { return max_tree_height_; }
 bool    set_max_tree_height( std::size_t new_height_) { 
     max_tree_height_ = std::min(new_height_, (std::size_t)MAX_TREE_HEIGHT); 
     return (new_height_ <= MAX_TREE_HEIGHT);
 }

 template< typename Row_index_iterator>
 void build_random_tree( Row_index_iterator begin, Row_index_iterator end, 
                         Dataset& dataset, Output& output, tree& t, typename tree::node& n, 
                        std::size_t height=0){
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
    if( height > max_tree_height() || std::distance(begin, end) < std::log( dataset.m())){
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
    ++height; //make sure to increment height!
    build_random_tree( left_indices.begin(), left_indices.end(), dataset, output, t, std::get<0>(kids), height);
    build_random_tree( right_indices.begin(), right_indices.end(), dataset, output, t, std::get<1>(kids),height);
 }

 template< typename Column_major_dataset>
 void train( Column_major_dataset& dataset, Output& output){
    trees.reserve( number_of_trees_);
    for( int i = 0; i < number_of_trees_; ++i){
        auto& current_tree = insert();
        //current_tree.reserve( dataset.m());
        auto& root = current_tree.insert_root();
        std::vector< std::size_t> row_indices( counting_iterator( 0),  counting_iterator( dataset.m()));
        build_random_tree( row_indices.begin(), row_indices.end(), dataset, output, current_tree, root); 
    }
 }

 template< typename Datapoint>
 Label_type classify( Datapoint& p) const{
     typedef std::unordered_map< Label_type, std::size_t> Map; 
     Map votes;
     for( auto & tree: trees){ votes[ tree.vote( p)]++; }
     typedef typename Map::value_type pair;
     std::max( votes.begin(), votes.end(), [&](const pair& a, const pair& b){ return a.second < b.second; });
 }
 
private:
 tree& insert(){
    trees.emplace_back();
    return trees.back();
 }
 std::vector< tree> trees;
 std::size_t number_of_trees_;
 std::size_t column_subset_size_;
 std::size_t max_tree_height_;
}; //end class random_forest

} //end namespace ml
} //end namespace ayasdi

