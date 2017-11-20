#ifndef CMAESAGENT_H
#define CMAESAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"

#include "cmaes_interface.h"

using namespace std;

/*
  Implement cmaes method. Input parameters: Population size,
  numEvalEpisodes. Code resembles cross-entropy method, but calls
  cmaes code for generating new popoulation.
 */

class CMAESAgent : public Agent{

 private:

  int numFeatures;
  int numActions;

  int numWeights;

  int popSize;
  int currentIndex;

  int numTotalEvalEpisodes;
  int numEvalEpisodes;

  vector<double> bestWeights;
  double bestValue;

  cmaes_t evo; /* an CMA-ES type struct or "object" */
  double *arFunvals, *const*pop;
  double bestVal;

  // Linear function approximator -- one for each action.
  vector< vector<double> > weights;
  vector<double> values;

  void generatePopulation();

  int takeAction(const vector<double> &state, const vector<double> &w);


 public:

  CMAESAgent(const int &numFeatures, const int &numActions, const int &popSize, const int &evalEpisodes, const int &randomSeed);
  ~CMAESAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

  vector<double> getBestWeights();

  void setBestWeightsAndValue(const vector<double> &w, const double &v);

};

#endif //CMAESAGENT_H

