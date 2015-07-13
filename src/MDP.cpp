//============================================================================
// Name        : MDP.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include<map>
#include <boost/numeric/ublas/io.hpp>
#include "storage_adaptors.hpp"
#include "MDP.hpp"

using namespace boost::numeric::ublas;


//Constructor intializing all member variables
MDP::MDP(std::map<int,matrix<double> > at, matrix<double> ar,double d){
	this->actionTransitions=at;
	this->actionReward=ar;
	this->discount=d;
	this->numStates=ar.size1();
	this->numActions=at.size();
}

//reward for each state associated with this policy (given the action reward and transition matrix for this MDP)
vector<double> MDP::policyReward(matrix<double> policy){

	vector<double> v (this->actionReward.size1());

	for (unsigned i = 0; i < policy.size1 (); ++ i) {

		matrix_row<matrix<double> > policyRow (policy, i);
		matrix_row<matrix<double> > rewardRow (this->actionReward, i);

		v(i)=sum(element_prod(policyRow, rewardRow));
	}

	return v;
}

//transition matrix associated with this policy (given the transition matrix for this MDP)
matrix<double> MDP::policyTransitions(matrix<double> policy){

	matrix<double> ptp=zero_matrix<double> (3,3);

	for (unsigned i = 0; i < policy.size1 (); ++ i) {

		  for (std::map<int,matrix<double> >::iterator it=this->actionTransitions.begin(); it!=this->actionTransitions.end(); ++it){

			matrix_row<matrix<double> > atmRow(it->second,i);

			row(ptp,i)+=atmRow*policy(i,it->first);
		}

	}

	return ptp;
}

//Compute the value function associated with a given policy
vector<double> MDP::policyEvaluation(matrix<double> pTransProb, vector<double> pReward){

	vector<double> valueFunction=zero_vector<double> (pReward.size());

	return pReward+prod(pTransProb,valueFunction);
}


struct actionValue{
	int action;
	double value;
};

//Greedy policy improvement given the current policy's value function
matrix<double> MDP::policyImprovement(vector<double> valueFunction){

	//int numStates=this.
	//int numActions=actionTransMatrix.size();

	matrix<double> greedyPolicy=zero_matrix<double> (this->numStates,this->numActions);

	for(unsigned i=0; i<numStates;++i){

		actionValue greedyAction={-1,0.0};

	  for (std::map<int,matrix<double> >::iterator it=this->actionTransitions.begin(); it!=this->actionTransitions.end(); ++it){

		   double value=inner_prod(row(it->second,i),valueFunction)+this->actionReward(i,it->first);

		   if(value>greedyAction.value){
			   greedyAction.action=it->first;
			   greedyAction.value=value;
		   }
	  }

	  greedyPolicy(i,greedyAction.action)=1.0;

	}

	return greedyPolicy;
}



/*
int main()
{
	double policy[3][2] = {
		0.9,0.1,
		0.7,0.3,
		0.2,0.8
	};

	matrix<double> A(3, 2);
	A = make_matrix_from_pointer(policy);

	std::cout << "policy: " << A << std::endl;


	double reward[3][2] = {
			1,2,
			0,1,
			1,0
		};

		matrix<double> b(3, 2);
		b = make_matrix_from_pointer(reward);

	std::cout << "reward: " << b << std::endl;


	vector<double> pr=policyReward(A,b);


	std::cout << "policy reward: " << pr << std::endl;


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


	double pol[3][2] = {
						1,0,
						1,0,
						0,1
					};

		matrix<double> P(3, 2);
		P = make_matrix_from_pointer(pol);

	matrix<double> ptm=policyTransProb(P,ps);


std::cout << "policy trans matrix: " << ptm << std::endl;


vector<double> pE=policyEvaluation(ptm,pr);

std::cout<<"policy evaluation: "<<pE<<std::endl;

	return 0;
}
*/
