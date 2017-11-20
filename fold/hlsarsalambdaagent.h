#ifndef HLSARSALAMBDAAGENT_H
#define HLSARSALAMBDAAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"

using namespace std;


class HLSarsaLambdaAgent : public Agent{

 private:

  int numFeatures;
  int numActions;

  int numEpisodesDone;

  // Linear function approximator -- one for each action.
  // Only works for binary features.
  vector< vector<double> > weights;

  vector<double> lastState;
  int lastAction;
  double lastReward;

  double alphaInit;
  double lambda;

  double alpha, epsilon;

  double alphaStart, alphaEnd, alphaDecay, alphaK1, alphaK2;
  double epsilonStart, epsilonEnd, epsilonDecay, epsilonK1, epsilonK2;

  // Eligibility traces
  vector< vector<double> > eligibility;

  // Visitation counts
  vector< vector<double> > visits;

  void resetEligibilityAndVisits();

  double computeQ(const vector<double> &state, const int &action);
  int argMaxQ(const vector<double> &state);

  bool diverged;

 public:

  HLSarsaLambdaAgent(const int &numFeatures, const int &numActions, const double &lambda, const double &alphaInit, const double &epsInit, const int &randomSeed);
  ~HLSarsaLambdaAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

  vector<double> getWeights();

};

#endif 

