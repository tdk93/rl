#include "transferagent.h"

TransferAgent::TransferAgent(const int &numFeatures, const int &numActions, const double &lambda, const int &evalEpisodes, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;
  
  transferPointEpisodes = 10000;
  numEpisodesDone = 0;

  sarsaLambdaAgent = new SarsaLambdaAgent(numFeatures, numActions, lambda, gsl_rng_get(ran));
  ceAgent = new CrossEntropyAgent(numFeatures, numActions, 100, evalEpisodes, gsl_rng_get(ran));

  numData = 0;
  
  // Boolean flag that becomes true once the switch from Sarsa to CE is effected. 
  transferred = false;

  diverged = false;
}

TransferAgent::~TransferAgent(){

  delete sarsaLambdaAgent;
  delete ceAgent;
}


int TransferAgent::takeAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  int action;

  if(numEpisodesDone < transferPointEpisodes){
    action = sarsaLambdaAgent->takeAction(state);
  }
  else{
    action = ceAgent->takeAction(state);
  }

  return action;
}

int TransferAgent::takeBestAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  int action;

  if(numEpisodesDone < transferPointEpisodes){
    action = sarsaLambdaAgent->takeBestAction(state);
  }
  else{
    action = ceAgent->takeBestAction(state);
  }

  return action;
}


void TransferAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

  if(diverged){
    return;
  }

  if(numEpisodesDone < transferPointEpisodes){
    sarsaLambdaAgent->update(reward, state, terminal);
  }
  else{
    ceAgent->update(reward, state, terminal);
  }

  if(terminal){
    numEpisodesDone++;
  }

  if(!transferred && (numEpisodesDone == transferPointEpisodes)){
    transfer();
  }
}


void TransferAgent::transfer(){

  vector<double> initMean, initVariance;

  initMean = sarsaLambdaAgent->getWeights();
  for(unsigned int i = 0; i < initMean.size(); i++){
    initVariance.push_back(1.0);

    if(isinf(initMean[i]) || (initMean[i] != initMean[i])){
      diverged = true;
    }
  }

  ceAgent->setMeanAndVariance(initMean, initVariance);

  transferred = true;
}

