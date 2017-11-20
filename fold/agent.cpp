#include "agent.h"

Agent::Agent(const int &numFeatures, const int &numActions, const int &randomSeed){

  ran = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(ran, randomSeed);
}


Agent::~Agent(){

  gsl_rng_free(ran);
}

