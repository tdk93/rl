#ifndef HILLCLIMBINGAGENT_H
#define HILLCLIMBINGAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"

using namespace std;

/*
  Generate popSize neighbours using Gaussian perturbations. Evaluate
  each, and update to highest one. If no progress, decay SD by epsilon.
 */


class HillClimbingAgent : public Agent{

 private:

  int numFeatures;
  int numActions;

  int numWeights;

  int popSize;

  double epsilon;

  int numTotalEvalEpisodes;
  int numEvalEpisodes;

  int currentIndex;

  vector<double> mean;
  vector<double> variance;
  
  vector<double> bestWeights;
  double bestValue;

  // Linear function approximator -- one for each action.
  vector< vector< double > > weights;
  vector<double> values;

  void generateWeights();

  int takeAction(const vector<double> &state, const vector<double> &w);


 public:

  HillClimbingAgent(const int &numFeatures, const int &numActions, const int &popSize, const int &evalEpisodes, const int &randomSeed);
  ~HillClimbingAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

  vector<double> getBestWeights();

  void setMeanAndvariance(const vector<double> &mean, const vector<double> &variance);
  void setBestWeightsAndValue(const vector<double> &w, const double &v);

};

#endif //HILLCLIMBINGAGENT_H

