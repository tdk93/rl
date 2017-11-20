#ifndef MDP_H
#define MDP_H

#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "headers.h"


using namespace std;


struct Cell{

  double x, y;
  bool isTerminal;

  // Features are binary. If a cell has a feature, cells with centres
  // with featureWidth/2 of its centre are activated when it is activated.
  bool hasFeature;
  int featureWidth;

  double reward;

  // Activations from every cell with a feature.
  vector<double> featureValue; // Array of 1's and 0's.
  vector<double> distance; // Distance to feature centre.

  // Value (and action to take from) under optimal policy.
  double maximalValue;
  int maximalAction;

  // Value (and action to take from) under least optimal policy.
  double minimalValue;
  int minimalAction;

};


class MDP{

 private:

  gsl_rng *ran;

  int side;
  int numTermini;
  int numStates;
  int currentState;

  int numFeatures;
  vector<Cell> cell;

  double actionNoiseProb;

  double stateNoiseWidthX;
  double stateNoiseWidthY;

  int minFeatureWidth;

  double lastReward;

  int numStepsThisEpisode;

  void fixTerminalStates();
  void fixRewards();

  int getNumCellsCovered();

  void fixFeatures();

  int getNeighbour(const int &s, const int &d);
  
  vector<double> computeMaximalDeterministicValue();
  vector<double> computeValue(const std::vector<int> &policy);
  vector<int> obtainPolicy(const std::vector<double> &value);
  void computeOptimalValueAndPolicy(const bool &maximal);

  
 public:

  MDP(const int &s, const double &p, const double &chi, const int &minFeatureWidth, const double &sigma, const int &randomSeed);
  ~MDP();

  void reset();

  double getMaximalValue();
  double getMinimalValue();
  double getValueRandomPolicy();

  int getNumFeatures();
  vector<double> getFeatures();
  double getLastReward();

  bool takeAction(const int &action);

  void display(const int &displayType);

};

#endif

