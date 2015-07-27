//============================================================================
// Name        : MDP.cpp
// Author      : Alex Minnaar
// Description : An c++ implementation of a Markov Decision Process (MDP)
//============================================================================
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include<map>
#include <boost/numeric/ublas/io.hpp>
#include "storage_adaptors.hpp"
#include "MDP.hpp"

using namespace boost::numeric::ublas;

//Constructor intializing all member variables
MDP::MDP(std::map<int, matrix<double> > at, matrix<double> ar, double d) {
	this->actionTransitions = at;
	this->actionReward = ar;
	this->discount = d;
	this->numStates = ar.size1();
	this->numActions = at.size();
}

//reward for each state associated with this policy (given the action reward and transition matrix for this MDP)
vector<double> MDP::policyReward(matrix<double> policy) {

	vector<double> v(this->actionReward.size1());

	for (unsigned i = 0; i < policy.size1(); ++i) {

		matrix_row<matrix<double> > policyRow(policy, i);
		matrix_row<matrix<double> > rewardRow(this->actionReward, i);

		v(i) = sum(element_prod(policyRow, rewardRow));
	}

	return v;
}

//transition matrix associated with this policy (given the transition matrix for this MDP)
matrix<double> MDP::policyTransitions(matrix<double> policy) {

	matrix<double> ptp = zero_matrix<double>(3, 3);

	for (unsigned i = 0; i < policy.size1(); ++i) {

		for (std::map<int, matrix<double> >::iterator it =
				this->actionTransitions.begin();
				it != this->actionTransitions.end(); ++it) {

			matrix_row<matrix<double> > atmRow(it->second, i);

			row(ptp, i) += atmRow * policy(i, it->first);
		}

	}

	return ptp;
}

//Compute the value function associated with a given policy
vector<double> MDP::policyEvaluation(matrix<double> pTransProb,
		vector<double> pReward) {

	vector<double> valueFunction = zero_vector<double>(pReward.size());

	return pReward + prod(pTransProb, valueFunction);
}

struct actionValue {
	int action;
	double value;
};

//Greedy policy improvement given the current policy's value function
matrix<double> MDP::policyImprovement(vector<double> valueFunction) {

	matrix<double> greedyPolicy = zero_matrix<double>(this->numStates,
			this->numActions);

	for (unsigned i = 0; i < numStates; ++i) {

		actionValue greedyAction = { -1, 0.0 };

		for (std::map<int, matrix<double> >::iterator it =
				this->actionTransitions.begin();
				it != this->actionTransitions.end(); ++it) {

			double value = inner_prod(row(it->second, i), valueFunction)
					+ this->actionReward(i, it->first);

			if (value > greedyAction.value) {
				greedyAction.action = it->first;
				greedyAction.value = value;
			}
		}

		greedyPolicy(i, greedyAction.action) = 1.0;

	}

	return greedyPolicy;
}
