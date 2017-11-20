#include "opttransferagent.h"

OptTransferAgent::OptTransferAgent(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const double &initWeight, const double &lambda, const double &alphaInit, const double &epsInit, const unsigned long int &transferPointEpisodes, const int &generations, const int &evalEpisodes, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;
  
  this->totalEpisodes = totalEpisodes;

  this->initWeight = initWeight;
  this->lambda = lambda;
  this->alphaInit = alphaInit;
  this->epsInit = epsInit;
  sarsaLambdaAgent = new SarsaLambdaAgent(numFeatures, numActions, totalEpisodes, initWeight, lambda, alphaInit, epsInit, gsl_rng_get(ran));

  this->transferPointEpisodes = transferPointEpisodes;

  this->generations = generations;
  this->evalEpisodes = evalEpisodes;
  //  optcmaesAgent = new OptCMAESAgent(numFeatures, numActions, totalEpisodes, generations, evalEpisodes, gsl_rng_get(ran));

  numEpisodesDone = 0;

  // Boolean flag that becomes true once the switch from Sarsa to CE is effected. 
  transferred = false;

  diverged = false;
}

OptTransferAgent::~OptTransferAgent(){

  delete sarsaLambdaAgent;
  if(transferred){
    delete optcmaesAgent;
  }
}


int OptTransferAgent::takeAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  int action;

  if(!transferred){
    action = sarsaLambdaAgent->takeAction(state);
  }
  else{
    action = optcmaesAgent->takeAction(state);
  }

  return action;
}

int OptTransferAgent::takeBestAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  int action;

  if(!transferred){
    action = sarsaLambdaAgent->takeBestAction(state);
  }
  else{
    action = optcmaesAgent->takeBestAction(state);
  }

  return action;
}


void OptTransferAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

  if(diverged){
    return;
  }

  if(!transferred){
    sarsaLambdaAgent->update(reward, state, terminal);
  }
  else{
    optcmaesAgent->update(reward, state, terminal);
  }
  
  if(terminal){
    numEpisodesDone++;
  }
  
  if(!transferred && (numEpisodesDone == transferPointEpisodes)){

    int numEpisodesLeft = totalEpisodes - transferPointEpisodes;

    if((generations * evalEpisodes) < numEpisodesLeft){
      transfer();
    }

  }
}


void OptTransferAgent::transfer(){

  vector<double> initMean, initVariance;

  initMean = sarsaLambdaAgent->getWeights();
  for(unsigned int i = 0; i < initMean.size(); i++){
    //    initVariance.push_back(1.0);

    if(isinf(initMean[i]) || (initMean[i] != initMean[i])){
      diverged = true;
    }
  }

  int n = initMean.size();
  double sum = 0;
  double sumSquared = 0;
  for(unsigned int i = 0; i < initMean.size(); i++){
    sum += initMean[i];
    sumSquared += initMean[i] * initMean[i];
  }

  double weightVar = ((n * sumSquared) - (sum * sum)) / (n * (n - 1));
  cout << "weightVar: " << weightVar;

  initVariance.resize(initMean.size(), weightVar);

  optcmaesAgent = new OptCMAESAgent(numFeatures, numActions, (totalEpisodes - transferPointEpisodes), generations, evalEpisodes, gsl_rng_get(ran), initMean, initVariance);

  transferred = true;
}

