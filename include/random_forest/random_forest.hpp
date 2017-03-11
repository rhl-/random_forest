#pragma once

//Project
#include <random_forest/decision_tree.hpp>
#include <random_forest/random_sample.hpp>

//STL
#include <unordered_map>
#include <cmath> //log
#include <numeric> //iota
#include <unordered_set> //set for oob error.
#include <set> //set for oob error.
#include <algorithm>

//BOOST
#include <boost/iterator/counting_iterator.hpp>

#define MAX_TREE_HEIGHT 32

namespace ml{
class random_forest{
public:
 typedef typename Output::value_type Label_type;
 typedef decision_tree< Label_type> tree;

 //Parametrized Constructor
 random_forest(std::size_t n_estimators=10, 
	       std::string criterion="gini", 
	       std::size_t max_depth=0, 
	       std::size_t min_samples_split=2, 
	       std::size_t min_samples_leaf=1, 
	       std::size_t min_weight_fraction_leaf=0.0, 
	       std::size_t max_features="auto", 
	       std::size_t max_leaf_nodes=0, 
	       double min_impurity_split=1e-07, 
	       bool bootstrap=true, 
	       bool oob_score=false,
	       int random_seed=0, 
	       int verbose=0){}

 random_forest(std::size_t number_of_trees, double column_subset_size, std::size_t max_tree_depth) :
 number_of_trees_(number_of_trees), column_subset_size_(column_subset_size), max_tree_depth_(max_tree_depth){}
 
private:
 typedef std::array< std::size_t, 2> Map;

 template< typename Row_index_iterator>
 bool is_pure_column( Row_index_iterator begin, Row_index_iterator end, Output& output){
     const auto& first_v = output[ *begin];
     return std::all_of(begin, end, [&]( const auto& i ){ return r==output[i]; } );
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
    std::fill( votes_.begin(), votes_.end(), 0);
    for( ; begin != end; ++begin){ votes_[ output[ *begin]]++; }

     typedef typename Map::value_type pair;
     auto max_elt= std::max_element( votes_.begin(), votes_.end()); 
     return std::distance( votes_.begin(), max_elt);
 }

 template< typename Row_index_iterator>
 Label_type get_majority_vote( Row_index_iterator begin, Row_index_iterator end, 
                               Row_index_iterator oob_begin, Row_index_iterator oob_end,
                               Output& output){
    std::fill( votes_.begin(), votes_.end(), 0);
    for( ; begin != end; ++begin){ votes_[ output[ *begin]]++; }
    for( ; oob_begin != oob_end; ++oob_begin){ votes_[ output[ *oob_begin]]++; }

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
    auto cmp = [&](const std::size_t& a, const std::size_t& b)->bool{ return (*(col_begin+a) < *(col_begin+b));};
    std::sort( row_idx_begin, row_idx_end, cmp);

    std::fill( lower_counts.begin(), lower_counts.end(), 0);
    std::fill( upper_counts.begin(), upper_counts.end(), 0);
    for( auto i = row_idx_begin; i != row_idx_end; ++i) { 
        upper_counts[ output_begin[ *i]]++; 
    }

    std::size_t number_of_rows=std::distance(row_idx_begin,row_idx_end);
    std::pair< std::size_t, double> best_split(0, std::numeric_limits< double>::infinity());
    //By assumption at this point number_of_rows > 1
    best_split.second = balanced_entropy( lower_counts, upper_counts, 
                                          1, number_of_rows-1, number_of_rows);
    for( auto split_index = row_idx_begin+1; split_index != row_idx_end;  ++split_index){
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
    }
    return best_split;
 }                                                                                                                 
                                                                                                                   
public:
 std::size_t max_tree_depth() const                   { return max_tree_depth_; }
 bool    set_max_tree_depth( std::size_t new_height_) {
     max_tree_depth_ = std::min(new_height_, (std::size_t)MAX_TREE_HEIGHT); 
     return (new_height_ <= MAX_TREE_HEIGHT);
 }

template< typename Vector>
void random_subset_size_k( std::size_t lower_bound, std::size_t upper_bound, std::size_t k, Vector& vector){
   typedef boost::counting_iterator< std::size_t> counting_iterator;

   vector.resize( k, 0);
   random_sample( counting_iterator( lower_bound), counting_iterator( upper_bound), 
                  vector.begin(), vector.end(), gen);
}

template< typename Vector>
void random_shuffle_range( std::size_t lower_bound, std::size_t upper_bound, Vector& vector){
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
 template< typename Row_index_iterator, typename Confusion_matrix>
 void build_random_tree( Row_index_iterator row_begin, Row_index_iterator row_end,
                         Row_index_iterator oob_begin, Row_index_iterator oob_end, 
                         Confusion_matrix& confusion_matrix,
                         Dataset& dataset, Output& output, tree& t, typename tree::node& n, 
                        std::size_t height=0){
    typedef std::vector< std::size_t> Vector;
    
    //Not possible to split, decision is already made.
    //Create a leaf node with this decision
    if( is_pure_column( row_begin, row_end, output)){
        generate_leaf_node(n, output[ *row_begin]);
        //Update OOB Confusion Matrix
        for( auto i = oob_begin; i != oob_end; ++i){ 
            confusion_matrix( output[ *row_begin] , output[ *i])++; 
        }
        return;
    }

    //TODO: Check if assuming log( number of data elements) is appropriate
    //Data is too small to waste time splitting. We punt.
    //Create a leaf node and give it a majority decision
    if( height >= max_tree_depth() || std::distance(row_begin, row_end) < std::log( dataset.m())){
        auto class_label = get_majority_vote( row_begin, row_end, output);
        //Update OOB Confusion Matrix
        for( auto i = oob_begin; i != oob_end; ++i){ 
            confusion_matrix( class_label, output[ *i])++; 
        }
        generate_leaf_node(n, class_label);
        return;
    }        

    //Choose a random subset of subset_size columns
    std::size_t subset_size = std::ceil(column_subset_size_*dataset.n());
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
                                                    row_begin, row_end, 
                                                    output.begin(), output.end());
        if( split_and_entropy.second < best_entropy){
            //Record the entropy so far and which column we are in
            best_entropy = split_and_entropy.second;
            column_index_for_split = column;

            //Index into sorted range of split.
            std::size_t split_index = split_and_entropy.first;

            //Get the entry containing the split_threshold_value
            split_threshold_value = *(dataset.begin( column)+row_begin[ split_index]);

            //Since we resort at every step we make physical copies of the row indices
            //If the column was gaurunteed sorted then we could skip this!
            //This seems like it should save a lot of memory..!
            row_indices_for_split = std::move( std::make_tuple( Vector( row_begin, row_begin+split_index),
                                                                Vector( row_begin+split_index, row_end)));
        }
    }
    //Build the split into the tree
    set_split( n, column_index_for_split, split_threshold_value);
    auto oob_middle = std::partition( oob_begin, oob_end, 
                    [&](const std::size_t& a){ return output[ a] < split_threshold_value; });
    //add children nodes into Decision Tree
    auto kids = t.insert_children( n);
    //Recursively call.
    auto& left_indices = std::get< 0>(row_indices_for_split);
    auto& right_indices = std::get< 1>(row_indices_for_split);
    ++height; //make sure to increment height!
    if(left_indices.size()) {
        build_random_tree(left_indices.begin(), left_indices.end(), 
                          oob_begin, oob_middle,
                          confusion_matrix,
                          dataset, output, t, 
                          std::get<0>(kids), height);
    }
    if( right_indices.size()) {
        auto& right_child = t.insert_right_child( n);
        build_random_tree(right_indices.begin(), right_indices.end(), 
                          oob_middle, oob_end,
                          confusion_matrix,
                          dataset, output, t, 
                          std::get<1>(kids), height);
    }
 }

 template< typename Column_major_dataset, typename Matrix>
 void fit( Column_major_dataset& dataset, Matrix& output){
    typedef std::vector< typename Column_major_dataset::value_type > Vector;
    trees.reserve( number_of_trees_);
    Matrix< int> confusion_matrix( 2, 2);
    ayasdi::timer t;
    double tree_time=0.0;
    Vector row( dataset.n(), 0.0);
    for( int i = 0; i < number_of_trees_; ++i){
        auto& current_tree = insert();
        current_tree.reserve( dataset.n());
        auto& root = current_tree.insert_root();
        std::vector< std::size_t> row_indices;
        random_shuffle_range(0, dataset.m(), row_indices);
        double row_fraction = .63;
        int row_subset_size = std::ceil( row_fraction*dataset.m());
                           //In Bag Points
        build_random_tree( row_indices.begin(), row_indices.begin() + row_subset_size,
                           //Out of Bag Points
                           row_indices.begin() + row_subset_size, row_indices.end(), 
                           confusion_matrix, //We will update this as we go.
                           dataset, output, current_tree, root); 
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
 std::size_t max_tree_depth_;
 std::random_device rd;
 std::mt19937 gen;
 Map votes_;
 Map lower_counts, upper_counts;
}; //end class random_forest

} //end namespace ml
