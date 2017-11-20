#ifndef OPTTRANSFERAGENT_H
#define OPTTRANSFERAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"
#include "sarsalambdaagent.h"
#include "optcmaesagent.h"

using namespace std;


class OptTransferAgent : public Agent{

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
  OptCMAESAgent *optcmaesAgent;

  bool transferred;

  void transfer();

  bool diverged;

 public:

  OptTransferAgent(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const double &initWeight, const double &lambda, const double &alphaInit, const double &epsInit, const unsigned long int &transferPointEpisodes, const int &generations, const int &evalEpisodes, const int &randomSeed);
  ~OptTransferAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

};

#endif 

