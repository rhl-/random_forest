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



#define MAX_TREE_HEIGHT 32

namespace ml{

struct rf_train_params{
 std::size_t n_estimators=10;
 std::string criterion="gini";
 std::size_t max_depth=0;
 std::size_t min_samples_split=2;
 std::size_t min_samples_leaf=1;
 std::size_t min_weight_fraction_leaf=0.0;
 std::size_t max_features="auto";
 std::size_t max_leaf_nodes=0;
 double min_impurity_split=1e-07;
 bool bootstrap=true;
 std::size_t row_fraction_size=.63;
 bool oob_score=false;
 int random_seed=0;
 int verbose=0;
};

template< typename Label_type>
class random_forest_classifier : public std::vector< tree> {
public:
 typedef decision_tree< Label_type> tree;
 random_forest(const rf_train_params& p): params( p) { insert_first_tree(); }
private:
 typedef std::vector< std::size_t> Map;

public:
 auto params() const                   { return params; }

 template< typename Datapoint>
 Label_type predict( Datapoint& p) const{
    for(auto& tree: (*this)){ votes[ tree.predict( p)]++; }
    typedef typename Map::value_type pair;
    auto max_elt=std::max_element( votes.begin(), votes.end());
    return std::distance( votes.begin(), max_elt);
 }
 
 template< typename Datapoint>
 std::tuple< Label_type, double> predict_proba( Datapoint& p) const{
    for(auto& tree: (*this)){ votes += tree.predict_proba( p); }
    typedef typename Map::value_type pair;
    auto max_elt=std::max_element( votes.begin(), votes.end());
    return std::make_tuple(std::distance( votes.begin(), max_elt), *max_elt);
 }
 
 void n_classes( std::size_t& n_classes_){ votes.resize( n_classes_); }

 tree& insert_next_tree(){
    trees.emplace_back();
    return trees.back();
 }
 
 rf_train_params params;
 std::random_device rd;
 std::mt19937 gen;
 Map votes;
}; //end class random_forest

} //end namespace ml

