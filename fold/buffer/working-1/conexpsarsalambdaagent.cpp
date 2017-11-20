#include "conexpsarsalambdaagent.h"

ConExpSarsaLambdaAgent::ConExpSarsaLambdaAgent(const int &numFeatures, const int &numActions, const double &lambda, const double &alphaStart, const double &epsilonStart, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){
  
  this->numFeatures = numFeatures;
  this->numActions = numActions;

  // Set all weights to 0 initially.
  for(int a = 0; a < numActions; a++){

    vector<double> w;
    for(int f = 0; f < numFeatures; f++){
      w.push_back(0);
    }

    weights.push_back(w);
  }

  for(int f = 0; f < numFeatures; f++){
    lastState.push_back(0);
  }
  lastAction = -1;
  lastReward = 0;

  numEpisodesDone = 0;

  this->lambda = lambda;

  // Clear history trajectory.
  resetEligibility();


  // Initial values for alpha and epsilon, which are
  // annealed harmonically in update() every 50,000
  // episodes.


  this->alphaInit = 0.1;
  alpha = alphaInit;
  epsilon = 0.1;

  int episodes = 50000;

  this->alphaStart = alphaStart;
  alphaEnd = 0.0001;
  alphaDecay = pow((alphaEnd / alphaStart), 1.0 / (episodes - 1));
  alpha = alphaStart;
  alphaK2 = (episodes - 1) / (alphaStart - alphaEnd);
  alphaK1 = alphaStart * alphaK2;

  this->epsilonStart = epsilonStart;
  epsilonEnd = 0.0001;
  epsilonDecay = pow((epsilonEnd / epsilonStart), 1.0 / (episodes - 1));
  epsilon = epsilonStart;
  epsilonK2 = (episodes - 1) / (epsilonStart - epsilonEnd);
  epsilonK1 = epsilonStart * epsilonK2;

  diverged = false;

}

ConExpSarsaLambdaAgent::~ConExpSarsaLambdaAgent(){
  
}


void ConExpSarsaLambdaAgent::resetEligibility(){

  eligibility.resize(numActions);
  for(int a = 0; a < numActions; a++){
    eligibility[a].resize(numFeatures);

    for(int f = 0; f < numFeatures; f++){
      eligibility[a][f] = 0;
    }
  }
}


double ConExpSarsaLambdaAgent::computeQ(const vector<double> &state, const int &action){

  double v = 0;
  
  for(int f = 0; f < numFeatures; f++){
    v += weights[action][f] * state[f];
  }

  return v;
}

int ConExpSarsaLambdaAgent::argMaxQ(const vector<double> &state){

  int bestAction = 0;
  double bestVal = computeQ(state, 0);

  int numTies = 0;

  for(int a = 1; a < numActions; a++){

    double val = computeQ(state, a);
    if(fabs(val - bestVal) < EPS){

      numTies++;
      if(gsl_rng_uniform(ran) < (1.0 / (1.0 + numTies))){
	bestVal = val;
	bestAction = a;
      }
    }
    else if(val > bestVal){
      bestVal = val;
      bestAction = a;
      numTies = 0;
    }
  }

  return bestAction;
}


int ConExpSarsaLambdaAgent::takeAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  for(int a = 0; a < numActions; a++){
    for(int f = 0; f < numFeatures; f++){

      if(isinf(weights[a][f]) || (weights[a][f] != weights[a][f])){
	diverged = true;
	//cout << "Diverged\n";
      }

      //if(gsl_rng_uniform(ran) < 1.0e-5){
      //cout << weights[a][f] << "\n";
      //}
    }
  }
  


  if(lastAction != -1){
    
    double nextQ = 0;
    for(int a = 0; a < numActions; a++){
      double q = computeQ(state, a);
      if(a == argMaxQ(state)){
	nextQ += (1.0 - epsilon + (epsilon / numActions)) * q;
      }
      else{
	nextQ += epsilon * q;
      }
    }

    double delta = lastReward + nextQ - computeQ(lastState, lastAction);

    for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
	eligibility[a][f] *= lambda;
      }
    }

    for(int f = 0; f < numFeatures; f++){

      eligibility[lastAction][f] += lastState[f]; // Must be zero or one.
      if(eligibility[lastAction][f] > 1.0){
      	eligibility[lastAction][f] = 1.0;
      }

    }

    double denominator = 0;
    for(int f = 0; f < numFeatures; f++){
      denominator += lastState[f];
    }

    for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
	weights[a][f] += alpha * delta * eligibility[a][f] * (1.0 / denominator);
      }
    }
  }

  int action;

  if(gsl_rng_uniform(ran) < epsilon){
    action = (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }
  else{
    action = argMaxQ(state);
  }

  for(int i = 0; i < numFeatures; i++){
    lastState[i] = state[i];
  }

  lastAction = action;

  return action;
}

int ConExpSarsaLambdaAgent::takeBestAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  return (argMaxQ(state));
}


void ConExpSarsaLambdaAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

  if(diverged){
    return;
  }

  lastReward = reward;

  if(terminal){

    double delta = lastReward - computeQ(lastState, lastAction);

    for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
	eligibility[a][f] *= lambda;
      }
    }
    for(int f = 0; f < numFeatures; f++){

      eligibility[lastAction][f] += lastState[f]; // Must be zero or one.
      if(eligibility[lastAction][f] > 1.0){
	eligibility[lastAction][f] = 1.0;
      }

    }

    double denominator = 0;
    for(int f = 0; f < numFeatures; f++){
      denominator += lastState[f];
    }

    for(int a = 0; a < numActions; a++){
      for(int f = 0; f < numFeatures; f++){
	weights[a][f] += alpha * delta * eligibility[a][f] * (1.0 / denominator);
      }
    }

    resetEligibility();

    lastAction = -1;

    numEpisodesDone++;


    /////////    // Anneal alpha and epsilon.
    /////////    alpha = alphaInit / ((numEpisodesDone / 50000.0) + 1.0);
    ////////    epsilon = 0.1 / ((numEpisodesDone / 50000.0) + 1.0);

    /////////////////////////
    /////////////////////////
    /////////////////////////
    /////////////////////////
    alpha = alphaK1 / (alphaK2 + numEpisodesDone - 1);
    alpha = alphaStart;

    epsilon = epsilonK1 / (epsilonK2 + numEpisodesDone - 1);
    epsilon = epsilonStart;
    /////////////////////////
    /////////////////////////
    /////////////////////////
    
  }

}

vector<double> ConExpSarsaLambdaAgent::getWeights(){

  vector<double> w;
  for(int a = 0; a < numActions; a++){
    for(int f = 0; f < numFeatures; f++){
      w.push_back(weights[a][f]);
    }
  }

  return w;
}

