#ifndef RANDOMAGENT_H
#define RANDOMAGENT_H

#include "agent.h"

/*
  Pick an action uniformly randomly.
 */

class RandomAgent : public Agent{

 private:

  int numActions;

 public:

  RandomAgent(const int &numFeatures, const int &numActions, const int &randomSeed);
  ~RandomAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

};

#endif //RANDOMAGENT_H

