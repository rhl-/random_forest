#include "catch.hpp"
#define NDEBUG

#include <iostream>
//Ayasdi
#include <ayasdi/decision_tree.hpp>
#include <ayasdi/random_forest.hpp>

namespace ml = ayasdi::ml;

typedef ml::decision_tree< double> tree; 
typedef tree::node tree_node;

TEST_CASE("Tree Tests", "[decision_tree]"){
 ml::decision_tree< double> t;
 auto r = t.insert_root();
 auto tree_root = t.root();
 SECTION("Tree tests"){
     SECTION("Insert Root Node"){
         REQUIRE(tree_root == r);
         REQUIRE(t.size() == 1);
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
         REQUIRE( &(t[  tree_root.left_child_index()]) == &left_child);
         REQUIRE( &(t[ tree_root.right_child_index()]) == &right_child);
     }
     SECTION("Insert Grand Children To The Left Child"){
         auto grand_children = t.insert_children( left_child);
         auto& grand_left_child  = std::get<0>( grand_children);
         auto& grand_right_child = std::get<1>( grand_children);
         REQUIRE(t[ t[ tree_root.left_child_index()].left_child_index()] == grand_left_child);
         REQUIRE(t[ t[ tree_root.left_child_index()].right_child_index()] == grand_right_child);
         SECTION("Constructors And Operators"){
             SECTION("Copy Constructor"){
                 ml::decision_tree< double> nt (t);
                 REQUIRE( nt.size() == t.size());
                 for( auto i=0; i< nt.size(); i++)
                 {
                     REQUIRE( nt[i] == t[i] );
                 }
                 SECTION("Equality operator") {
                     REQUIRE( nt == t );
                 }
                 SECTION("Move Constructor"){
                     ml::decision_tree<double> c( std::move( t));
                     REQUIRE( c == nt);
                     REQUIRE( t.size() == 0);
                 }
                 SECTION("Inequality Operator"){
                     ml::decision_tree<double> c;
                     REQUIRE( c != nt);
                 }
                 SECTION( "Assignment Operator"){
                     ml::decision_tree<double> c = t;
                     REQUIRE( c == t);
                 }
             }
         }
     }
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
        REQUIRE_FALSE( c != n);
    }
    SECTION( "Assignment Operator"){
    }
  
  }
}
