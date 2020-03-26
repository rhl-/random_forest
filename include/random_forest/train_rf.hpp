//
// Created by Ryan H. Lewis on 3/11/17.
//

#ifndef RANDOM_FOREST_TRAIN_RF_HPP
#define RANDOM_FOREST_TRAIN_RF_HPP

#include <random_forest.hpp>


template< typename Row_index_iterator>
bool is_pure_column( Row_index_iterator begin, Row_index_iterator end, Output& output){
 const auto& first_v = output[ *begin];
 return std::all_of(begin, end, [&]( const auto& i ){ return r==output[i]; } );
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


template< typename T>
class Matrix_view {
public:
 Matrix_view(T* raw_ptr, size_t rows, size_t cols):
 ptr(raw_ptr), n_rows( rows), n_cols( cols), data(rows*cols) {}
 T& operator()(size_t i, size_t j){ return ptr[j * n_cols + i];  }
 T operator()(size_t i, size_t j) const{ return (*this)(i,j); }
 std::size_t height()const { return n_rows; };
 std::size_t width()const { return n_cols; };
 auto begin(){ return ptr; }
 auto end(){ return ptr+n_rows*n_cols;}
 auto begin( std::size_t i){ return &((*this)(0,i)); }
 auto end( std::size_t i){ return begin(i)+n_rows; }
 auto data() { return data; }
private:
 size_t n_rows;
 size_t n_cols;
 T* ptr;
};

template< typename T>
class Matrix : public Matrix_view<T>{
public:
 Matrix(size_t rows, size_t cols): data(rows*cols) {
   Matrix_view::Matrix_view(data.data(), rows, cols);
 }
private:
 std::vector<T> data;
};


template< typename T, typename O>
Matrix<int> fit( Matrix_view<T>& dataset, Matrix_view<O>& output, rf_train_params params=rf_train_params()){
 typedef std::vector< decltype(dataset(0,0))> vector;
 trees.reserve( number_of_trees_);
 Matrix< int> confusion_matrix( 2, 2);
 vector row( dataset.height(), 0.0);
 random
 for( int i = 0; i < number_of_trees_; ++i){
  auto& current_tree = rf.insert_next_tree();
  current_tree.reserve( dataset.width());
  auto& root = current_tree.insert_root();
  std::vector< std::size_t> row_indices;
  random_shuffle_range(0, dataset.m(), row_indices);
  int row_subset_size = std::ceil( params.row_fraction_size*dataset.height());
  //In Bag Points
  auto row_begin = row_indices.begin();
  auto row_end = row_indices.begin() + row_subset_size;
  build_random_tree(  row_begin, row_end,
                      //Out of Bag Points
                     row_end, row_indices.end(),
                     confusion_matrix, //We will update this as we go.
                     dataset, output, current_tree, root);
 }
 return confusion_matrix;
}

#endif //RANDOM_FOREST_TRAIN_RF_HPP
