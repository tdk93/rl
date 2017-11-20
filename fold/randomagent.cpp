#include "randomagent.h"

RandomAgent::RandomAgent(const int &numFeatures, const int &numActions, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){

  this->numActions = numActions;
}

RandomAgent::~RandomAgent(){
}


int RandomAgent::takeAction(const vector<double> &state){

  return ((int)(gsl_rng_uniform(ran) * numActions) % numActions);
}

int RandomAgent::takeBestAction(const vector<double> &state){

  return ((int)(gsl_rng_uniform(ran) * numActions) % numActions);
}

void RandomAgent::update(const double &reward, const vector<double> &state, const bool &terminal){
  // Do nothing. No learning by random agent.
}

