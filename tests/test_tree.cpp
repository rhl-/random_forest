#include "catch.hpp"

#include <iostream>
//Ayasdi
#include <ayasdi/decision_tree.hpp>
#include <ayasdi/random_forest.hpp>

namespace ml = ayasdi::ml;

TEST_CASE("Tree Tests", "[decision_tree]"){
 ml::decision_tree< double> t;
 auto tree_root = t.root();
 SECTION("Tree tests"){
     SECTION("Insert Root Node"){
      auto r = t.insert_root();
      REQUIRE(tree_root == r);
     }
      auto children = t.insert_children( tree_root);
      auto& left_child = std::get<0>( children);
      auto& right_child = std::get<1>( children);
     SECTION("Insert Children"){
      REQUIRE( t[  tree_root.left_child_index()] == left_child);
      REQUIRE( t[ tree_root.right_child_index()] == right_child);
     }
     SECTION("Insert Children to the Existing Node"){
      auto children_again = t.insert_children( tree_root);
      REQUIRE( &t[  tree_root.left_child_index()] == &left_child);
      REQUIRE( &t[ tree_root.right_child_index()] == &right_child);
     }
     SECTION("Insert Grand Children To The Left Child"){
      auto grand_children = t.insert_children( left_child);
      auto& grand_left_child  = std::get<0>( grand_children);
      auto& grand_right_child = std::get<1>( grand_children);
      REQUIRE(t[ t[ tree_root.left_child_index()].left_child_index()] == grand_left_child);
      REQUIRE(t[ t[ tree_root.left_child_index()].right_child_index()] == grand_right_child);
     }
 }
}
