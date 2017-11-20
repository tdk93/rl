#ifndef CMAESAGENT_H
#define CMAESAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"
#include "cmaes_interface.h"

using namespace std;

class CMAESAgent : public Agent{

 private:

  int numFeatures;
  int numActions;

  int popSize;
  int numWeights;
  int selectSize;

  cmaes_t evo;
  //  double *arFunvals, *const*pop, *xfinal;

  int currentIndex;
  int numTotalEvalEpisodes;
  int numEvalEpisodes;

  //  vector<double> mean;
  //  vector<double> variance;


  vector<double> bestWeights;
  double bestValue;

  // Linear function approximator -- one for each action.
  // Contains a bias term (the last one).
  vector< vector<double> > weights;
  vector<double> values;

  void generatePopulation();
  //  void computeNextMeanAndVariance();

  int takeAction(const vector<double> &state, const vector<double> &w);


 public:

  CMAESAgent(const int &numFeatures, const int &numActions, const int &randomSeed);
  ~CMAESAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

  vector<double> getBestWeights();

  void setMeanAndvariance(const vector<double> &mean, const vector<double> &variance);
  void setBestWeightsAndValue(const vector<double> &w, const double &v);

};

#endif //CMAESAGENT_H

