#ifndef TRANSFERAGENT_H
#define TRANSFERAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"
#include "sarsalambdaagent.h"
#include "crossentropyagent.h"

using namespace std;


class TransferAgent : public Agent{

 private:

  int numFeatures;
  int numActions;

  int transferPointEpisodes;

  int numEpisodesDone;

  vector<Transition> data;
  Transition currentTransition;
  int numData;

  SarsaLambdaAgent *sarsaLambdaAgent;
  CrossEntropyAgent *ceAgent;

  bool transferred;

  void transfer();

  bool diverged;

 public:

  TransferAgent(const int &numFeatures, const int &numActions, const double &lambda, const int &evalEpisodes, const int &randomSeed);
  ~TransferAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

};

#endif 
