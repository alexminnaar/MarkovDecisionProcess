/*
 * MDPTest.cpp
 *
 *  Created on: Jul 12, 2015
 *      Author: alexminnaar
 */
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../MDP.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include<map>
#include "../storage_adaptors.hpp"
#include <boost/numeric/ublas/io.hpp>


using namespace boost::numeric::ublas;

unsigned int Factorial( unsigned int number ) {
    return number <= 1 ? number : Factorial(number-1)*number;
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
    REQUIRE( Factorial(1) == 1 );
    REQUIRE( Factorial(2) == 2 );
    REQUIRE( Factorial(3) == 6 );
    REQUIRE( Factorial(10) == 3628800 );
}


TEST_CASE("action policy matrix is computed","[policyTransitions]"){

	double p1[3][3] = {
			0.2,0.3,0.5,
			0.7,0.1,0.2,
			0.5,0.4,0.1
		};

	matrix<double> P1(3, 3);
	P1 = make_matrix_from_pointer(p1);


	double p2[3][3] = {
					0.6,0.2,0.2,
					0.3,0.3,0.4,
					0.1,0.1,0.8
				};

	matrix<double> P2(3, 3);
	P2 = make_matrix_from_pointer(p2);

	std::map<int,matrix<double> > ps;

	ps[0]=P1;
	ps[1]=P2;

	std::cout<<P2<<std::endl;

}

