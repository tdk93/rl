#include "doubleqlearningagent.h"

DoubleQLearningAgent::DoubleQLearningAgent(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const double &initWeight,  const double &lambda, const double &alphaStart, const double &epsilonStart, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){
  
  this->numFeatures = numFeatures;
  this->numActions = numActions;
  curQ = 0;

  // Set all weights to 0 initially.
  // Set all weights to initWeight initially.
  
  for(int a = 0; a < numActions; a++){
    
    vector < vector<double> > x;

    for (int td = 0; td < 2; td++){
      vector< double> w;
      for(int f = 0; f < numFeatures; f++){
        w.push_back(initWeight);
      }

      x.push_back(w);
    }
    weights.push_back(x);
  }
  cout << "weights [0][0][0] ";
  cout << weights[0][0][0]<<endl;


  for(int f = 0; f < numFeatures; f++){
    lastState.push_back(0);
  }
  lastAction = -1;
  lastReward = 0;

  numEpisodesDone = 0;

  this->lambda = lambda;

  // Clear history trajectory.
  resetEligibility();


  this->alphaInit = 0.1;
  alpha = alphaInit;
  epsilon = 0.1;

  double alphaEnd = 0.01;
  alpha = alphaStart;
  alphaK2 = (alphaEnd * (totalEpisodes - 1)) / (alphaStart - alphaEnd);
  alphaK1 = alphaStart * alphaK2;

  double epsilonEnd = 0.01;
  epsilon = epsilonStart;
  epsilonK2 = (epsilonEnd * (totalEpisodes - 1)) / (epsilonStart - epsilonEnd);
  epsilonK1 = epsilonStart * epsilonK2;

  diverged = false;
}

DoubleQLearningAgent::~DoubleQLearningAgent(){
  
}


void DoubleQLearningAgent::resetEligibility(){

  eligibility.resize(numActions);
  for(int a = 0; a < numActions; a++){
    eligibility[a].resize(numFeatures);

    for(int f = 0; f < numFeatures; f++){
      eligibility[a][f] = 0;
    }
  }
}


double DoubleQLearningAgent::computeQ(const vector<double> &state, const int &action, bool qVal){

  double v = 0;
  
  for(int f = 0; f < numFeatures; f++){
    v += weights[action][qVal][f] * state[f];
  }

  return v;
}

double DoubleQLearningAgent::computeQSpecial(const vector<double> &state, const int &action, bool qVal){

  double v = 0;
  for(int f = 0; f < numFeatures; f++){
    v += (weights[action][qVal][f] + weights[action][~qVal][f]/2.0) * state[f];
  }

  return v;
}
/*
double DoubleQLearningAgent::computeQForBestAction(const vector<double> &state, const int &action, bool qVal){

  double v = 0;
  
  for(int f = 0; f < numFeatures; f++){
    v += weights[action][qVal][f] * state[f];
  }

  return v;
}
*/




int DoubleQLearningAgent::argMaxQForBestAction(const vector<double> &state){

  int bestAction = 0;
  double bestVal = -10000000000;
  
  int numTies = 0;

  for(int a = 1; a < numActions; a++){

    double val = computeQSpecial(state, a, 0);//+computeQ(state,a,1))/2.0;
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

int DoubleQLearningAgent::argMaxQ(const vector<double> &state, bool qVal){

  int bestAction = 0;
  double bestVal = computeQ(state,0,qVal);

  int numTies = 0;

  for(int a = 1; a < numActions; a++){

    double val = computeQ(state, a, qVal);
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


int DoubleQLearningAgent::takeAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  for(int a = 0; a < numActions; a++){
    for(int f = 0; f < numFeatures; f++){

      
      if(isinf(weights[a][curQ][f]) || (weights[a][curQ][f] != weights[a][curQ][f]) || isinf(weights[a][!curQ][f]) || (weights[a][!curQ][f] != weights[a][!curQ][f])){
        diverged = true;
        cout << isinf(weights[a][curQ][f]); 
        cout << (weights[a][curQ][f] != weights[a][curQ][f]) ; cout << isinf(weights[a][!curQ][f]);  cout << (weights[a][!curQ][f] != weights[a][!curQ][f]);
 
        if(true){
          cout << "a is " << a << endl;
          cout << "f is " << f << endl;
        }
        cout << "Diverged\n";
      }

    }
  }
  

  /*
  if(lastAction != -1){
    
    double delta = lastReward + computeQ(state, argMaxQ(state,curQ),!curQ) - computeQ(lastState, lastAction,curQ);

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
	weights[a][curQ][f] += alpha * delta * eligibility[a][f] * (1.0 / denominator);
      }
    }
  }
  */


  int action;

 if(gsl_rng_uniform(ran) < 0.5){
   curQ = 0;
  }
  else{
    curQ = 1;
  }


  if(gsl_rng_uniform(ran) < epsilon){
    action = (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }
  else{
    action = argMaxQ(state,curQ);
  }

  for(int i = 0; i < numFeatures; i++){
    lastState[i] = state[i];
  }

  lastAction = action;
  //curQ = !curQ
  return action;
}

int DoubleQLearningAgent::takeBestAction(const vector<double> &state){

  if(diverged){
    return (int)(gsl_rng_uniform(ran) * numActions) % numActions;
  }

  return (argMaxQForBestAction(state));
}


void DoubleQLearningAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

  if(diverged){
    return;
  }

  lastReward = reward;

  if(!terminal){
    double delta = lastReward + computeQ(state, argMaxQ(state,curQ),!curQ) - computeQ(lastState, lastAction,curQ);

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
	weights[a][curQ][f] += alpha * delta * eligibility[a][f] * (1.0 / denominator);
      }
    }

  }



  else{

  double delta = lastReward - computeQ(lastState, lastAction,curQ);

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
	weights[a][curQ][f] += alpha * delta * eligibility[a][f] * (1.0 / denominator);
      }
    }

    resetEligibility();

    lastAction = -1;

    numEpisodesDone++;


    alpha = alphaK1 / (alphaK2 + numEpisodesDone - 1);
    epsilon = epsilonK1 / (epsilonK2 + numEpisodesDone - 1);
  }

}

vector<double> DoubleQLearningAgent::getWeights(){

  cout << "how did we end up here"<<endl;
  vector<double> w;
  for(int a = 0; a < numActions; a++){
    for(int f = 0; f < numFeatures; f++){
      w.push_back(weights[a][curQ][f]);
    }
  }

  return w;
}

