#pragma once

//Project
#include <ayasdi/decision_tree.hpp>
#include <ayasdi/random_sample.hpp>
#include <ayasdi/math.hpp>

//STL
#include <unordered_map>
#include <cmath> //log
#include <numeric> //iota
#include <unordered_set> //set for oob error.
#include <set> //set for oob error.

#define MAX_TREE_HEIGHT 30

namespace ayasdi{
namespace ml{

template< typename Dataset, typename Output>
class random_forest{
public:
 typedef typename Output::value_type Label_type;
 typedef decision_tree< Label_type> tree;

 //Default Constructor
 random_forest() : number_of_trees_(500), column_subset_size_(0.30), max_tree_height_(MAX_TREE_HEIGHT)
 {}
 random_forest( const random_forest& f): number_of_trees_(f.number_of_trees_), 
                                         column_subset_size_(f.column_subset_size_), 
                                         max_tree_height_(f.max_tree_height_){}
 //Parametrized Constructor
 random_forest(std::size_t number_of_trees, double column_subset_size, std::size_t max_tree_height) :
 number_of_trees_(number_of_trees), column_subset_size_(column_subset_size), max_tree_height_(max_tree_height)
 {}
 
private:
 typedef std::array< std::size_t, 2> Map;

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
 void generate_leaf_node( typename tree::node& n, Label_type label){ 
    n.split_value_ = label; 
 }
 
 //Set data of node to be an internal decision node of tree
 void set_split( typename tree::node& n, std::size_t& column_index, double& split_threshold_value){
     n.split_ = column_index;
     n.split_value_ = split_threshold_value;
 }
 
 template< typename Row_index_iterator>
 Label_type get_majority_vote( Row_index_iterator begin, Row_index_iterator end, Output& output){
    votes_.fill( 0);
    for( ; begin != end; ++begin){ votes_[ output[ *begin]]++; }

     typedef typename Map::value_type pair;
     auto max_elt= std::max_element( votes_.begin(), votes_.end()); 
     return std::distance( votes_.begin(), max_elt);
 }

 template< typename Counts>
 inline double entropy( const Counts& counts, std::size_t& n){
     double denominator = 1.0/n;
     double entropy=0.0;
     for( const auto& q: counts){
         const double p = (double)q*denominator;
         if( p > 0){ entropy -= p*std::log( p); }
      }
      return entropy;
 }

 template< typename Counts>
 inline double balanced_entropy( Counts& lower_counts, Counts& upper_counts, 
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
    //We just sort the row indices into order
    //We can GPU accelerate this for fun with thrust::sort()
    //Also we can try tbb::sort()
    ayasdi::timer t;
    auto cmp = [&](const std::size_t& a, const std::size_t& b)->bool{ return (*(col_begin+a) < *(col_begin+b));};
    std::sort( row_idx_begin, row_idx_end, cmp);

    lower_counts.fill( 0); upper_counts.fill( 0);
    for( auto i = row_idx_begin; i != row_idx_end; ++i) { 
        upper_counts[ output_begin[ *i]]++; 
    }

    std::size_t number_of_rows=std::distance(row_idx_begin,row_idx_end);
    std::pair< std::size_t, double> best_split(0, std::numeric_limits< double>::infinity());
    //By assumption at this point number_of_rows > 1
    best_split.second = balanced_entropy( lower_counts, upper_counts, 
                                          1, number_of_rows-1, number_of_rows);
    for( auto split_index = row_idx_begin+1; split_index != row_idx_end;  ++split_index){
    t.start();
        auto class_label = output_begin[ *split_index];
        lower_counts[class_label]++;
        upper_counts[class_label]--;
        //This logic handles repeated values in the input column
        if( col_begin+*split_index == col_begin+*(split_index-1)){ continue; }
        auto lower_index = std::distance(row_idx_begin,split_index)+1;
        auto upper_index = number_of_rows-lower_index;
        auto current_entropy = balanced_entropy( lower_counts, upper_counts, 
                                                 lower_index, upper_index, 
                                                 number_of_rows);
        if( current_entropy < best_split.second){     
           best_split.first  = std::distance( row_idx_begin, split_index);
           best_split.second = current_entropy;
           if( current_entropy == 0.0 ){ return best_split; }
        }
    t.stop();
    }
    return best_split;
 }                                                                                                                 
                                                                                                                   
public:
 std::size_t max_tree_height() const                   { return max_tree_height_; }
 bool    set_max_tree_height( std::size_t new_height_) {
     max_tree_height_ = std::min(new_height_, (std::size_t)MAX_TREE_HEIGHT); 
     return (new_height_ <= MAX_TREE_HEIGHT);
 }
template< typename Vector>
void random_subset_from_range( std::size_t lower_bound, std::size_t upper_bound, Vector& vector){
    std::mt19937 gen(rd());
    //std::uniform_int_distribution<> dis(lower_bound, upper_bound);
    //vector.reserve( .63*(upper_bound - lower_bound)); //(1-1/e)*range
    //for( std::size_t i = 0; i < (upper_bound-lower_bound); ++i){ 
    //    auto idx = dis(gen);
    //    auto it = std::lower_bound( vector.begin(), vector.end(), idx);
    //    if( it == vector.end() || ((it != vector.end()) && (*it != idx))){ vector.insert(it, idx); }
    //}
   vector.resize( upper_bound-lower_bound, 0);
   std::iota( vector.begin(), vector.end(), 0);
   std::random_shuffle( vector.begin(), vector.end());
}
 template< typename Row_index_iterator>
 void build_random_tree( Row_index_iterator row_begin, Row_index_iterator row_end, 
                         Dataset& dataset, Output& output, tree& t, typename tree::node& n, 
                        std::size_t height=0){
    typedef std::vector< std::size_t> Vector;
    
    //Not possible to split, decision is already made.
    //Create a leaf node with this decision
    if( is_pure_column( row_begin, row_end, output)){
        generate_leaf_node(n, output[ *row_begin]);
        return;
    }

    //TODO: Check if assuming log( number of data elements) is appropriate
    //Data is too small to waste time splitting. We punt.
    //Create a leaf node and give it a majority decision
    if( height >= max_tree_height() || std::distance(row_begin, row_end) <= 5){//std::log( dataset.m())){
        auto class_label = get_majority_vote( row_begin, row_end, output);
        generate_leaf_node(n, class_label);
        return;
    }        

    //Choose a random subset of subset_size columns
    std::size_t subset_size = column_subset_size_*dataset.n();
    //create a vector of length n
    Vector columns;
    random_subset_from_range(0, dataset.n(), columns);
    columns.erase( columns.begin()+column_subset_size_*(dataset.n()), columns.end());

    //Output Variables
    double best_entropy=std::numeric_limits< double>::infinity();
    std::size_t column_index_for_split=std::numeric_limits< std::size_t>::infinity();
    double split_threshold_value=best_entropy;
    std::tuple< Vector, Vector> row_indices_for_split;
    //Find the best split within each column, find minimal overall split.
    for(auto& column: columns){
        std::pair< std::size_t, double> 
        split_and_entropy = find_best_column_split( dataset.begin( column), dataset.end( column), 
                                                    row_begin, row_end, //Obs: We may sort this all we like.
                                                    output.begin(), output.end());
        if( split_and_entropy.second < best_entropy){
            //Record the entropy so far and which column we are in
            best_entropy = split_and_entropy.second;
            column_index_for_split = column;

            //Index into sorted range of split.
            std::size_t split_index = split_and_entropy.first;

            //Get the entry containing the split_threshold_value
            split_threshold_value = *(dataset.begin( column)+*(row_begin+split_index));

            //Since we resort at every step we make physical copies of the row indices
            //If the column was gaurunteed sorted then we could skip this!
            //This seems like it should save a lot of memory..!
            row_indices_for_split = std::move( std::make_tuple( Vector( row_begin, row_begin+split_index),
                                                                Vector( row_begin+split_index, row_end)));
        }
    }
    //Build the split into the tree
    set_split( n, column_index_for_split, split_threshold_value);
    //add children nodes into Decision Tree
    auto kids = t.insert_children( n);
    //Recursively call.
    auto& left_indices = std::get< 0>(row_indices_for_split);
    auto& right_indices = std::get< 1>(row_indices_for_split);
    ++height; //make sure to increment height!
    if(left_indices.size()) {
        build_random_tree(left_indices.begin(), left_indices.end(), dataset, output, t, 
                          std::get<0>(kids), height);
    }
    if( right_indices.size()) {
        auto& right_child = t.insert_right_child( n);
        build_random_tree(right_indices.begin(), right_indices.end(), dataset, output, t, 
                          std::get<1>(kids), height);
    }
 }

 template< typename Column_major_dataset>
 void train( Column_major_dataset& dataset, Output& output){
    typedef math::Vector< typename Column_major_dataset::value_type > Vector;
    trees.reserve( number_of_trees_);
    math::Matrix< int> confusion_matrix( 2, 2);
    ayasdi::timer t;
    double tree_time=0.0;
    Vector row( dataset.n(), 0.0);
    for( int i = 0; i < number_of_trees_; ++i){
        auto& current_tree = insert();
        current_tree.reserve( dataset.n());
        auto& root = current_tree.insert_root();
        std::vector< std::size_t> row_indices;
        random_subset_from_range(0, dataset.m(), row_indices);
        double row_fraction = .63;
        t.start();
        build_random_tree( row_indices.begin(), row_indices.begin() + row_fraction*dataset.m(), dataset, output, current_tree, root); 
        t.stop();
        tree_time += t.elapsed();

        Map votes;
        for( auto j = 0; j < dataset.n(); ++j){ 
            for( auto i = row_indices.begin() + row_fraction*dataset.m(); 
                      i != row_indices.end(); ++i){ row[ j] = dataset( *i, j); }
        }
        for( auto i = row_indices.begin() + row_fraction*dataset.m(); i != row_indices.end(); ++i){
            auto label = classify( row, votes);
            confusion_matrix( label , output[ *i])++;
            votes.fill( 0);
        }
    }
    std::cout << "Tree Time: " << tree_time << std::endl;
    std::cout << "Confusion Matrix: " << confusion_matrix << std::endl;

    auto success = confusion_matrix(0,0) + confusion_matrix(1,1);
    auto failure = confusion_matrix(1,0) + confusion_matrix(0,1);
    auto total = success+failure;
    std::cout << "OOB Error: " << (1.0 - ((double)success/total)) << std::endl;
 }

 template< typename Datapoint>
 Label_type classify( Datapoint& p) const{
    Map votes;
    return classify( p, votes);
 }
 
private:
 template< typename Datapoint>
 Label_type classify( Datapoint& p, Map& votes) const{
    for(auto& tree: trees){ votes[ tree.vote( p)]++; }
    typedef typename Map::value_type pair;
    auto max_elt= 
    std::max_element( votes.begin(), votes.end());
    return std::distance( votes.begin(), max_elt);
 }

 tree& insert(){
    trees.emplace_back();
    return trees.back();
 }

 std::vector< tree> trees;
 std::size_t number_of_trees_;
 double column_subset_size_;
 std::size_t max_tree_height_;
 std::random_device rd;
 Map votes_;
 Map lower_counts, upper_counts;
}; //end class random_forest

} //end namespace ml
} //end namespace ayasdi

