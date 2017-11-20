#ifndef RWGAGENT_H
#define RWGAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"

using namespace std;

/*
  Repeatedly sample weights from static mean and variance -- Gaussian
  distribution. Evaluate for some number of episodes, and update best
  set if performance is highest.
 */


class RWGAgent : public Agent{

 private:

  int numFeatures;
  int numActions;

  int numWeights;

  int numTotalEvalEpisodes;
  int numEvalEpisodes;

  vector<double> mean;
  vector<double> variance;
  
  vector<double> bestWeights;
  double bestValue;

  // Linear function approximator -- one for each action.
  vector< double > weights;
  double value;

  void generateWeights();

  int takeAction(const vector<double> &state, const vector<double> &w);


 public:

  RWGAgent(const int &numFeatures, const int &numActions, const int &evalEpisodes, const int &randomSeed);
  ~RWGAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

  vector<double> getBestWeights();

  void setMeanAndvariance(const vector<double> &mean, const vector<double> &variance);
  void setBestWeightsAndValue(const vector<double> &w, const double &v);

};

#endif //RWGAGENT_H

