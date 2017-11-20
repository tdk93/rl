#ifndef PSHCAGENTSTEEP_H
#define PSHCAGENTSTEEP_H

#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "Agent.h"

#define MAX_FEATURES 500
#define MAX_ACTIONS 4
#define MAX_POP_SIZE 50

using namespace std;

class PSHCAgentSteep : public Agent{

 private:

  int numFeatures;
  int numActions;

  int popSize;

  double weights[MAX_POP_SIZE][MAX_ACTIONS][MAX_FEATURES];
  double values[MAX_POP_SIZE];
  
  double currentWeights[MAX_ACTIONS][MAX_FEATURES];
  double currentValue;
  
  int numTotalEvalEpisodes;
  int numEvalEpisodes;
  double weightDelta;
  double deltaDecayFraction;

  int numEvaluated;

  void generatePopulation();


 public:
  
  PSHCAgentSteep(int numFeatures, int numActions);
  ~PSHCAgentSteep();

  int takeAction(double state[]);
  int takeBestAction(double state[]);
  void update(double reward, double state[], bool terminal);

};

#endif //PSHCAGENTSTEEP_H
