#ifndef PSHCAGENT_H
#define PSHCAGENT_H

#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "Agent.h"

#define MAX_FEATURES 500
#define MAX_ACTIONS 4

using namespace std;

class PSHCAgent : public Agent{

 private:

  int numFeatures;
  int numActions;

  int numWorseEvals;
  int maxWorseEvals;

  double currentWeights[MAX_ACTIONS][MAX_FEATURES];
  double currentValue;

  double nextWeights[MAX_ACTIONS][MAX_FEATURES];
  double nextValue;
  
  int numTotalEvalEpisodes;
  int numEvalEpisodes;

  double maxWeightDelta;
  double currentWeightDelta;
  double deltaDecayFraction;

  void generateNext();


 public:
  
  PSHCAgent(int numFeatures, int numActions);
  ~PSHCAgent();

  int takeAction(double state[]);
  int takeBestAction(double state[]);
  void update(double reward, double state[], bool terminal);

};

#endif //PSHCAGENT_H
