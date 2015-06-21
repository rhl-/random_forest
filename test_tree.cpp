#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <iostream>
#include "tree.hpp"

namespace ml = ayasdi::ml;

TEST_CASE("Tree Tests", "[decision_tree]"){
 ml::decision_tree t;
 auto g = t.root();
 SECTION("Root Nodes"){
  auto r = t.insert_root();
  REQUIRE(g == r);
 }
 SECTION("Children"){
  auto children = t.insert_children( g);
  auto& left_child = std::get<0>( children);
  auto& right_child = std::get<1>( children);
  REQUIRE( t[  g.left_child()] == left_child);
  REQUIRE( t[ g.right_child()] == right_child);
 }
}
