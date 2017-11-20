#ifndef TRANSFERAGENT_H
#define TRANSFERAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"
#include "sarsalambdaagent.h"
#include "cmaesagent.h"

using namespace std;


class TransferAgent : public Agent{

 private:

  int numFeatures;
  int numActions;

  unsigned long int totalEpisodes;

  int transferPointEpisodes;

  int numEpisodesDone;

  double initWeight;
  double lambda;
  double alphaInit;
  double epsInit;
  SarsaLambdaAgent *sarsaLambdaAgent;

  int generations;
  int evalEpisodes;
  CMAESAgent *cmaesAgent;

  bool transferred;

  void transfer();

  bool diverged;

 public:

  TransferAgent(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const double &initWeight, const double &lambda, const double &alphaInit, const double &epsInit, const unsigned long int &transferPointEpisodes, const int &generations, const int &evalEpisodes, const int &randomSeed);
  ~TransferAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

};

#endif 

