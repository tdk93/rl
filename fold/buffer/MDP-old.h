#ifndef MDP_H
#define MDP_H

#include <time.h>
#include <stdio.h>
#include <iostream.h>
#include <stdlib.h>
#include <math.h>

#define MAX_STATES 10000
#define TERMINI_FRACTION 0.05

#define DIR_NORTH 0
#define DIR_SOUTH 1
#define DIR_EAST 2
#define DIR_WEST 3

#define ACTION_NORTH 0
#define ACTION_SOUTH 1
#define ACTION_EAST 2
#define ACTION_WEST 3

using namespace std;

class MDP{

 private:

  int side;
  int numTermini;
  int numStates;
  int currentState;
  bool terminal[MAX_STATES];

  int numFeatures;
  int feature[MAX_STATES];

  double stateNoiseSigma;
  double actionNoiseProb;

  double lastReward;

  void fixTerminalStates();
  void fixFeatures();

  int getNeighbour(int s, int d);

 public:

  MDP(int side, double featureFraction, double stateNoiseSigma, double actionNoiseProb);
  ~MDP();

  void reset();

  int getNumFeatures();
  void getFeatures(double f[]);
  double getLastReward();

  bool takeAction(int action);

  void display();

  double computeDeterministicOptimalValue();
};

#endif
