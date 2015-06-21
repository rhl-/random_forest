#pragma once

#include "tree.hpp"
#include <unordered_map>

namespace ayasdi{
namespace ml{

template< typename Dataset, typename Output>
class random_forest{
public:
 typedef typename Output::value_type Label_type;
private:
 typedef decision_tree< Label_type> tree;
 void train( Dataset& dataset, Output& output){
     //TODO: All the work should be here.
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
 std::vector< tree > trees;
}; //end class random_forest

} //end namespace ml
} //end namespace ayasdi

