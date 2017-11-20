#ifndef MDP_H
#define MDP_H

#include <time.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <math.h>


#define TERMINI_FRACTION 0.01
#define FEATURE_SD 0.5

#define DIR_NORTH 0
#define DIR_SOUTH 1
#define DIR_EAST 2
#define DIR_WEST 3

#define ACTION_NORTH 0
#define ACTION_SOUTH 1
#define ACTION_EAST 2
#define ACTION_WEST 3

#define DISPLAY_FEATURES 0
#define DISPLAY_TERMINAL 1
#define DISPLAY_REWARDS  2
#define DISPLAY_VALUES  3
#define DISPLAY_ACTIONS 4

using namespace std;

struct Cell{

  double x, y;
  bool isTerminal;
  bool hasFeature;
  double reward;
  double optimalValue;
  int optimalAction;
  std::vector<double> featureValue;
};

class MDP{

 private:

  int side;
  int numTermini;
  int numStates;
  int currentState;

  int numFeatures;
  std::vector<Cell> cell;

  double stateNoiseSigma;
  double actionNoiseProb;

  double lastReward;

  void fixTerminalStates();
  void fixRewards();
  void fixFeatures();

  int getNeighbour(int s, int d);
  
  std::vector<double> computeOptimalDeterministicValue();
  std::vector<double> computeValue(std::vector<int> policy);
  std::vector<int> obtainPolicy(std::vector<double> value);
  void computeOptimalValueAndPolicy();
  
 public:

  MDP(int side, double featureFraction, double stateNoiseSigma, double actionNoiseProb);
  ~MDP();

  void reset();

  double getOptimalValue();

  int getNumFeatures();
  std::vector<double> getFeatures();
  double getLastReward();

  bool takeAction(int action);

  void display(int displayType);

};

#endif
