#ifndef CONSARSALAMBDAAGENT_H
#define CONSARSALAMBDAAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"

using namespace std;


class ConSarsaLambdaAgent : public Agent{

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

  void resetEligibility();

  double computeQ(const vector<double> &state, const int &action);
  int argMaxQ(const vector<double> &state);

  bool diverged;

 public:

  ConSarsaLambdaAgent(const int &numFeatures, const int &numActions, const double &lambda, const double &alphaInit, const double &epsInit, const int &randomSeed);
  ~ConSarsaLambdaAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

  vector<double> getWeights();

};

#endif 

