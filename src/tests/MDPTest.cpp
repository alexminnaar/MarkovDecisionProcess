/*
 * MDPTest.cpp
 *
 * Test cases for the MDP implementation.
 *
 *  Created on: Jul 12, 2015
 *      Author: alexminnaar
 */
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../MDP.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include<map>
#include "../storage_adaptors.hpp"
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

//function creating an MDP object for testing
MDP createTestMDP(void) {

	//create test probability transition matrix
	double p1[3][3] = { 0.2, 0.3, 0.5, 0.7, 0.1, 0.2, 0.5, 0.4, 0.1 };

	matrix<double> P1(3, 3);
	P1 = make_matrix_from_pointer(p1);

	double p2[3][3] = { 0.6, 0.2, 0.2, 0.3, 0.3, 0.4, 0.1, 0.1, 0.8 };

	matrix<double> P2(3, 3);
	P2 = make_matrix_from_pointer(p2);

	std::map<int, matrix<double> > ps;

	ps[0] = P1;
	ps[1] = P2;

	//create test reward matrix
	double reward[3][2] = { 1, 2, 0, 1, 1, 0 };

	matrix<double> b(3, 2);
	b = make_matrix_from_pointer(reward);

	double discount = 0.5;

	//create MDP object
	MDP myMDP = MDP(ps, b, discount);

	return myMDP;
}

TEST_CASE("action policy matrix is computed","[policyTransitions]") {

	MDP myMDP = createTestMDP();

	//create test policy
	double policy[3][2] = { 0.9, 0.1, 0.7, 0.3, 0.2, 0.8 };

	matrix<double> A(3, 2);
	A = make_matrix_from_pointer(policy);

	//compute policy transition matrix
	matrix<double> policyTrans = myMDP.policyTransitions(A);

	//computed policy transition matrix should equal the following gold standard
	double correct[3][3] = { 0.24, 0.29, 0.47, 0.58, 0.16, 0.26, 0.18, 0.16,
			0.66 };

	matrix<double> correctPT(3, 3);
	correctPT = make_matrix_from_pointer(correct);

	REQUIRE(correctPT.size1() == policyTrans.size1());

	for (int i = 0; i < correctPT.size1(); ++i) {
		for (int j = 0; j < correctPT.size2(); ++j) {

			//truncate for minor differences
			REQUIRE(
					floor(policyTrans(i, j) * 10) / 10
							== floor(correctPT(i, j) * 10) / 10);
		}
	}
}

TEST_CASE("policy reward is computed","[policyReward]") {

	MDP myMDP = createTestMDP();

	//create test policy
	double policy[3][2] = { 0.9, 0.1, 0.7, 0.3, 0.2, 0.8 };

	matrix<double> A(3, 2);
	A = make_matrix_from_pointer(policy);

	//compute policy reward vector
	vector<double> policyRew = myMDP.policyReward(A);

	//test against gold standard policy reward vector
	double correct[3] = { 1.1, 0.3, 0.2 };

	for (int i = 0; i < 3; ++i) {
		REQUIRE(correct[i] == policyRew[i]);
	}
}

TEST_CASE("the bellman equation is evaluated","[policyEvaluation]") {

	MDP myMDP = createTestMDP();

	//create test policy
	double policy[3][2] = { 0.9, 0.1, 0.7, 0.3, 0.2, 0.8 };
	matrix<double> A(3, 2);
	A = make_matrix_from_pointer(policy);

	//compute policy transition matrix
	matrix<double> policyTrans = myMDP.policyTransitions(A);

	//compute policy reward vector
	vector<double> policyRew = myMDP.policyReward(A);

	//create an initial value function needed for Bellman equation
	double valueFunction[] = { 0.3, 0.2, 1.1 };
	vector<double> vf(3);
	vf = make_vector_from_pointer(3, valueFunction);

	//compute result of Bellman equation
	vector<double> bellman = myMDP.bellmanEquation(policyTrans, policyRew, vf);

	//correct Bellman equation result
	double correctVf[] = { 1.42, 0.54, 0.6 };

	for (int i = 0; i < 3; ++i) {
		//truncate for minor differences
		REQUIRE(
				floor(correctVf[i] * 100) / 100
						== floor(bellman[i] * 100) / 100);
	}

}

TEST_CASE("a policy is evaluated","[policyEvaluation]") {

	MDP myMDP = createTestMDP();

	//create test policy
	double policy[3][2] = { 0.9, 0.1, 0.7, 0.3, 0.2, 0.8 };
	matrix<double> A(3, 2);
	A = make_matrix_from_pointer(policy);

	//compute policy transition matrix
	matrix<double> policyTrans = myMDP.policyTransitions(A);

	//compute policy reward vector
	vector<double> policyRew = myMDP.policyReward(A);

	//evaluated value function for policy
	vector<double> vFunc = myMDP.policyEvaluation(policyTrans, policyRew,
			0.001);

	//correct value function
	double correct[] = { 1.56313, 0.905347, 0.615901 };

	for (int i = 0; i < 3; i++) {
		//truncate for minor differences
		REQUIRE(floor(correct[i] * 100) / 100 == floor(vFunc(i) * 100) / 100);
	}

}

TEST_CASE("a policy is improved","[policyImprovement]") {

	MDP myMDP = createTestMDP();

	double vf[] = { 1.5, 0.9, 0.6 };

	vector<double> valueFunction(3);
	valueFunction = make_vector_from_pointer(3, vf);

	matrix<double> improvedPolicy = myMDP.policyImprovement(valueFunction);

	//correct improved policy
	double correct[3][2] = { 0.0, 1.0, 0.0, 1.0, 1.0, 0.0 };

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			REQUIRE(improvedPolicy(i, j) == correct[i][j]);
		}
	}

}

TEST_CASE("full policy iteration algorithm","[policyIteration]"){

	MDP myMDP = createTestMDP();

	matrix<double> optimalPolicy=myMDP.policyIteration();

	std::cout<<optimalPolicy<<std::endl;

	//correct improved policy
	double correct[3][2] = { 0.0, 1.0, 0.0, 1.0, 1.0, 0.0 };

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			REQUIRE(optimalPolicy(i, j) == correct[i][j]);
		}
	}

}
