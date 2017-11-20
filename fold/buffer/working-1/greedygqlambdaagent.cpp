#include "greedygqlambdaagent.h"

GreedyGQLambdaAgent::GreedyGQLambdaAgent(const int &numFeatures, const int &numActions, const double &lambda, const double &alphaStart, const double &epsilonStart, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){
  
  this->numFeatures = numFeatures;
  this->numActions = numActions;

  // Set all weights to 0 initially.
  w.clear();
  theta.clear();
  e.clear();
  for(int a = 0; a < numActions; a++){

    vector<double> ww;
    vector<double> tt;
    vector<double> ee;
    for(int f = 0; f < numFeatures; f++){
      ww.push_back(0);
      tt.push_back(0);
      ee.push_back(0);
    }

    w.push_back(ww);
    theta.push_back(tt);
    e.push_back(ee);
  }

  for(int f = 0; f < numFeatures; f++){
    lastState.push_back(0);
  }
  lastAction = -1;
  lastExplore = false;

  lastReward = 0;

  numEpisodesDone = 0;

  this->lambda = lambda;

  // Clear history trajectory.
  resetEligibility();

  int episodes = 50000;

  double alphaEnd = 0.01;
  alpha = alphaStart;
  alphaK2 = (alphaEnd * (episodes - 1)) / (alphaStart - alphaEnd);
  alphaK1 = alphaStart * alphaK2;

  double epsilonEnd = 0.01;
  epsilon = epsilonStart;
  epsilonK2 = (epsilonEnd * (episodes - 1)) / (epsilonStart - epsilonEnd);
  epsilonK1 = epsilonStart * epsilonK2;

  diverged = false;
}

GreedyGQLambdaAgent::~GreedyGQLambdaAgent(){
  
}


void GreedyGQLambdaAgent::resetEligibility(){

  e.resize(numActions);
  for(int a = 0; a < numActions; a++){
    e[a].resize(numFeatures);

    for(int f = 0; f < numFeatures; f++){
      e[a][f] = 0;
    }
  }
}


double GreedyGQLambdaAgent::computeQ(const vector<double> &state, const int &action){

  double v = 0;
  
  for(int f = 0; f < numFeatures; f++){
    v += theta[action][f] * state[f];
  }

  return v;
}

int GreedyGQLambdaAgent::argMaxQ(const vector<double> &state){

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


double GreedyGQLambdaAgent::dot(vector<double> v1, vector<double> v2){

    double sum = 0;

    for(unsigned int i = 0; i < v1.size(); i++){
        sum += v1[i] * v2[i];
    }

    return sum;
}

int GreedyGQLambdaAgent::takeAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  for(int a = 0; a < numActions; a++){
    for(int f = 0; f < numFeatures; f++){


      if(isinf(theta[a][f]) || (theta[a][f] != theta[a][f])){
	diverged = true;
	cout << "Diverged\n";
      }

    }
  }

  int bestAction = argMaxQ(state);
  
  if(lastAction != -1){
    
    double denominator = 0;
    for(int f = 0; f < numFeatures; f++){
      denominator += lastState[f];
    }
    
    for(int a = 0; a < numActions; a++){

      vector<double> phi = lastState;
      vector<double> phi_next = state;
      //lambda already set.
      double gamma = 1.0;
      double z = 0;
      double r = lastReward;
      double eta = 1.0;
      double rho = lastExplore? 0 : 1.0 / (1.0 - epsilon + (epsilon / numActions));//Does not matter.
      double I = 1.0;
      double delta = r + (1 - gamma) * z + gamma * dot(theta[bestAction], phi_next) - dot(theta[lastAction], phi);

      if(a != bestAction){
	phi_next.resize(state.size(), 0);
      }
      if(a != lastAction){
	phi.resize(state.size(), 0);
      }
      
      for(int i = 0; i < numFeatures; i++){
	e[a][i] = rho * e[a][i] + I * phi[i];
      }

      for(int i = 0; i < numFeatures; i++){
	
	theta[a][i] += alpha * (delta * e[a][i] - 0 * gamma * (1.0 - lambda) * dot(w[a], e[a]) * phi_next[i]) * (1.0 / denominator);
	w[a][i] += alpha * eta * (delta * e[a][i] - dot(w[a], phi) * phi[i]) * (1.0 / denominator);
	e[a][i] *= gamma * lambda;
      }

    }

  }
  
  int action = bestAction;
  bool explore = false;
  if(gsl_rng_uniform(ran) < epsilon){
    action = (int)(gsl_rng_uniform(ran) * numActions) % numActions;
    if(action != bestAction){
      explore = true;
    }
  }

  for(int i = 0; i < numFeatures; i++){
    lastState[i] = state[i];
  }
  
  lastAction = action;
  lastExplore = explore;
  
  return action;
}

int GreedyGQLambdaAgent::takeBestAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  return (argMaxQ(state));
}


void GreedyGQLambdaAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

  if(diverged){
    return;
  }

  lastReward = reward;

  if(terminal){

    double denominator = 0;
    for(int f = 0; f < numFeatures; f++){
      denominator += lastState[f];
    }

    
    for(int a = 0; a < numActions; a++){

      vector<double> phi = lastState;
      vector<double> phi_next = state;

      //lambda already set.
      double gamma = 1.0;
      double z = 0;
      double r = lastReward;
      double eta = 1.0;
      double rho = lastExplore? 0 : 1.0 / (1.0 - epsilon + (epsilon / numActions));//Does not matter.
      double I = 1.0;
      double delta = r + (1 - gamma) * z - dot(theta[lastAction], phi);

      if(a != lastAction){
	phi.resize(state.size(), 0);
      }
      phi_next.resize(state.size(), 0);

      for(int i = 0; i < numFeatures; i++){
	e[a][i] = rho * e[a][i] + I * phi[i];
      }

      for(int i = 0; i < numFeatures; i++){
	
	theta[a][i] += alpha * (delta * e[a][i] - 0 * gamma * (1.0 - lambda) * dot(w[a], e[a]) * phi_next[i]) * (1.0 / denominator);
	w[a][i] += alpha * eta * (delta * e[a][i] - dot(w[a], phi) * phi[i]) * (1.0 / denominator);
	e[a][i] *= gamma * lambda;
      }
    }
    
    resetEligibility();
    
    lastAction = -1;
    lastExplore = false;

    numEpisodesDone++;


    alpha = alphaK1 / (alphaK2 + numEpisodesDone - 1);
    epsilon = epsilonK1 / (epsilonK2 + numEpisodesDone - 1);
  }

}

vector<double> GreedyGQLambdaAgent::getWeights(){

  vector<double> tt;
  for(int a = 0; a < numActions; a++){
    for(int f = 0; f < numFeatures; f++){
      tt.push_back(theta[a][f]);
    }
  }

  return tt;
}

