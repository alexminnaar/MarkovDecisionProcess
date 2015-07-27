/*
 * MDP.hpp
 *
 *	An c++ implementation of a Markov Decision Process (MDP)
 *
 *  Created on: Jul 12, 2015
 *      Author: alexminnaar
 */

#ifndef MDP_HPP_
#define MDP_HPP_

#include <boost/numeric/ublas/matrix.hpp>
#include<map>

using namespace boost::numeric::ublas;


class MDP{

private :
	//mapping from action to probability transition matrix
	std::map<int,matrix<double> > actionTransitions;

	//matrix where entry (i,j) is the reward associated with taking action j from state j
	matrix<double> actionReward;

	//MDP discount factor in [0,1]
	double discount;

	//Total number of states in MPD
	int numStates;

	//Total number of actions in MDP
	int numActions;

public:

	//Constructor intializing all member variables
	MDP(std::map<int,matrix<double> > at, matrix<double> ar, double d);

	//reward for each state associated with this policy (given the action reward and transition matrix for this MDP)
	vector<double> policyReward(matrix<double> policy);

	//transition matrix associated with this policy (given the transition matrix for this MDP)
	matrix<double> policyTransitions(matrix<double> policy);

	//Compute the value function associated with a given policy
	vector<double> policyEvaluation(matrix<double> policyTrans, vector<double> policyRew);

	//Greedy policy improvement given the current policy's value function
	matrix<double> policyImprovement(vector<double> valueFunction);

};

#endif /* MDP_HPP_ */
