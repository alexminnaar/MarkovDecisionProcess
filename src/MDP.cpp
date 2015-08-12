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

//Constructor initializing all member variables
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

//Compute the result of the Bellman equation
vector<double> MDP::bellmanEquation(matrix<double> policyTrans,
		vector<double> policyRew, vector<double> valueFunc) {

	return policyRew + this->discount * prod(policyTrans, valueFunc);
}

//Compute the value function associated with a given policy
vector<double> MDP::policyEvaluation(matrix<double> pTransProb,
		vector<double> pReward, double epsilon) {

	//Initialize value function to zero
	vector<double> valueFunction = zero_vector<double>(pReward.size());

	double delta = 10.0;

	while (delta > epsilon) {

		vector<double> previousValueFunction = valueFunction;

		//compute value function via the Bellman equation
		valueFunction = bellmanEquation(pTransProb, pReward, valueFunction);

		//Check for convergence
		vector<double> diffVect = valueFunction - previousValueFunction;

		//get maximum element
		vector<double>::iterator result;

		//delta is most changed element (if it is small then convergence has occurred)
		delta = *std::max_element(diffVect.begin(), diffVect.end());
	}

	return valueFunction;
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

			//compute the value associated with this action at this state
			double value = inner_prod(row(it->second, i), valueFunction)
					+ this->actionReward(i, it->first);

			//select the greedy action in terms of value
			if (value > greedyAction.value) {
				greedyAction.action = it->first;
				greedyAction.value = value;
			}
		}

		greedyPolicy(i, greedyAction.action) = 1.0;
	}

	return greedyPolicy;
}

//compute the optimal policy for this MDP (corresponding value function can be found using policyEvalution method)
matrix<double> MDP::policyIteration() {

	//initialize with a random policy
	matrix<double> currentPolicy = scalar_matrix<double>(this->numStates,
			this->numActions);

	currentPolicy = currentPolicy / (double) this->numActions;

	matrix<double> oldPolicy = zero_matrix<double>(this->numStates,
			this->numActions);

	const double epsilon = std::numeric_limits<double>::epsilon();

	while (!detail::equals(currentPolicy, oldPolicy, epsilon, epsilon)) {

		oldPolicy = currentPolicy;

		matrix<double> policyTrans = this->policyTransitions(currentPolicy);

		vector<double> policyReward = this->policyReward(currentPolicy);

		vector<double> policyValue = this->policyEvaluation(policyTrans,
				policyReward, 0.001);

		currentPolicy = this->policyImprovement(policyValue);
	}

	return currentPolicy;
}
