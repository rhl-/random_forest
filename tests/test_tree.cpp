#include "catch.hpp"

#include <iostream>
//Ayasdi
#include <ayasdi/decision_tree.hpp>
#include <ayasdi/random_forest.hpp>

namespace ml = ayasdi::ml;

TEST_CASE("Tree Tests", "[decision_tree]"){
 ml::decision_tree< double> t;
 auto g = t.root();
 SECTION("Root Nodes"){
  auto r = t.insert_root();
  REQUIRE(g == r);
 }
 SECTION("Children"){
  auto children = t.insert_children( g);
  auto& left_child = std::get<0>( children);
  auto& right_child = std::get<1>( children);
  REQUIRE( t[  g.left_child_index()] == left_child);
  REQUIRE( t[ g.right_child_index()] == right_child);
 }
}
