#ifndef OPTCMAESAGENT_H
#define OPTCMAESAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"

#include "cmaes_interface.h"

using namespace std;

/*
  Implement cmaes method with intelligent sampling. Input parameters:
  Population size, numEvalEpisodes. Code resembles cross-entropy
  method, but calls cmaes code for generating new popoulation.
 */


class OptCMAESAgent : public Agent{

 private:

  int numFeatures;
  int numActions;

  int numWeights;

  int popSize;
  int selectSize;
  int currentIndex;

  int numTotalEvalEpisodes;
  int numEvalEpisodes;

  vector<double> bestWeights;
  double bestValue;

  vector<double> mean;

  cmaes_t evo; /* an CMA-ES type struct or "object" */
  double *arFunvals, *const*pop;
  double bestVal;

  // Linear function approximator -- one for each action.
  vector< vector<double> > weights;
  vector< vector<double> > values;
  vector<double> meanValues;

  void generatePopulation();

  int takeAction(const vector<double> &state, const vector<double> &w);

  double currentEpisodeReward;

  int getIndexToSample();
  vector<bool> eliminated;
  int numEliminated;

 public:

  OptCMAESAgent(const int &numFeatures, const int &numActions,  const unsigned long int &totalEpisodes, const int &generations, const int &evalEpisodes, const int &randomSeed);
  OptCMAESAgent(const int &numFeatures, const int &numActions,  const unsigned long int &totalEpisodes, const int &generations, const int &evalEpisodes, const int &randomSeed, const vector<double> &startMean, const vector<double> &startVariance);

  ~OptCMAESAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

  vector<double> getBestWeights();

  void setBestWeightsAndValue(const vector<double> &w, const double &v);

};

#endif //OPTCMAESAGENT_H

