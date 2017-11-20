#ifndef MDP_H
#define MDP_H

#include <time.h>
#include <stdio.h>
#include <iostream.h>
#include <stdlib.h>
#include <math.h>

#include "Geometry.h"

#define TERMINI_FRACTION 0.05
#define NUM_TERMINAL_SQUARES 5

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

  VecPosition currentPosition;

  double terminalSquareSide;
  VecPosition terminalSquareCentres[NUM_TERMINAL_SQUARES];

  double strideLength;
  double stateNoiseSigma;
  double actionNoiseSigma;

  double lastReward;

  double normal(const double &mean, const double &std);
  void fixTerminal();
  
  bool isTerminal(VecPosition point, int terminalSquareIndex);
  bool isTerminal(VecPosition point);


 public:

  MDP(double strideLength, double stateNoiseSigma, double actionNoiseProb);
  ~MDP();

  void reset();

  void getState(double state[]);
  double getLastReward();
  bool takeAction(int action);

  void display();

};

#endif

