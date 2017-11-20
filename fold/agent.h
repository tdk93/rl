#ifndef AGENT_H
#define AGENT_H

#include <vector>

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "headers.h"

using namespace std;


struct Transition{

  vector<double> state;
  int action;
  double reward;
  vector<double> nextState;
  bool terminal;

};


class Agent{

 private:

 protected:

  gsl_rng *ran;

 public:

  Agent(const int &numFeatures, const int &numActions, const int &randomSeed);
  virtual ~Agent();

  virtual int takeAction(const vector<double> &state) = 0;
  virtual int takeBestAction(const vector<double> &state) = 0;
  virtual void update(const double &reward, const vector<double> &state, const bool &terminal) = 0;

};

#endif //AGENT_H

