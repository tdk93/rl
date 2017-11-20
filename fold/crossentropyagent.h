#ifndef CROSSENTROPYAGENT_H
#define CROSSENTROPYAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"

using namespace std;

/*
  Implement cross-entropy method. Input parameters: Population size,
  numEvalEpisodes.
 */


class CrossEntropyAgent : public Agent{

 private:

  int numFeatures;
  int numActions;

  int numWeights;

  int popSize;
  int selectSize;
  int currentIndex;

  int numTotalEvalEpisodes;
  int numEvalEpisodes;

  vector<double> mean;
  vector<double> variance;
  
  vector<double> bestWeights;
  double bestValue;

  // Linear function approximator -- one for each action.
  vector< vector<double> > weights;
  vector<double> values;

  void generatePopulation();
  void computeNextMeanAndVariance();

  int takeAction(const vector<double> &state, const vector<double> &w);


 public:

  CrossEntropyAgent(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const int &generations, const int &evalEpisodes, const int &randomSeed);
  ~CrossEntropyAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

  vector<double> getBestWeights();

  void setMeanAndVariance(const vector<double> &mean, const vector<double> &variance);
  void setBestWeightsAndValue(const vector<double> &w, const double &v);

};

#endif //CROSSENTROPYAGENT_H

