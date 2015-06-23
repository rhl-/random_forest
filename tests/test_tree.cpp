#include "catch.hpp"

#include <iostream>
//Ayasdi
#include <ayasdi/decision_tree.hpp>
#include <ayasdi/random_forest.hpp>

namespace ml = ayasdi::ml;

typedef ml::decision_tree< double> tree; 
typedef tree::node tree_node;

TEST_CASE("Tree Tests", "[decision_tree]"){
 tree t;
 auto g = t.root();
 SECTION("Insert Root Node"){
  auto r = t.insert_root();
  REQUIRE(g == r);
 }
 SECTION("Insert Children"){
  auto children = t.insert_children( g);
  auto& left_child = std::get<0>( children);
  auto& right_child = std::get<1>( children);
  REQUIRE( t[  g.left_child_index()] == left_child);
  REQUIRE( t[ g.right_child_index()] == right_child);
 }
}

TEST_CASE("Node Tests"){
    tree_node n;
    SECTION("Constructors"){
     SECTION("Default Constructor"){
        REQUIRE( n.split_ == 0);
        REQUIRE( n.split_value_ ==0);
        CHECK( n.is_leaf());
        CHECK_FALSE( n.is_not_leaf());
     }
     n.split_ = 3;
     n.split_value_ = 3.14;
     SECTION("Copy Constructor"){
         tree_node c( n);
         REQUIRE( c.split_ == n.split_);
         REQUIRE( c.split_value_ == n.split_value_);
         CHECK( c.is_leaf());
         CHECK_FALSE( c.is_not_leaf());
     }
     SECTION("Move Constructor"){
        tree_node c( std::move( n));
        REQUIRE( c.split_ == 3);
        REQUIRE( c.split_value_ == 3.14);
        CHECK( c.is_leaf());
        CHECK_FALSE( c.is_not_leaf());
    }
   }
   SECTION("Assignment and Equality Operators"){
    SECTION("Equality Operator"){
        tree_node c( n);
        REQUIRE(c == n);
    }
    SECTION("Inequality Operator"){
        tree_node c;
        REQUIRE( c != n);
    }
    SECTION( "Assignment Operator"){
    }
  
  }
}
