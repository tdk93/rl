#include "hillclimbingagent.h"

HillClimbingAgent::HillClimbingAgent(const int &numFeatures, const int &numActions, const int &popSize, const int &evalEpisodes, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;
  
  numWeights = numFeatures * numActions;

  this->popSize = popSize;
  this->numTotalEvalEpisodes = evalEpisodes;

  this->epsilon = 0.95;

  // Initialise means and variances to 0, 1 respectively.
  for(int i = 0; i < numWeights; i++){

    mean.push_back(0);
    variance.push_back(1.0);

    bestWeights.push_back(0);
  }

  bestValue = -INF;//minus infinity

  weights.resize(popSize);
  values.resize(popSize);
  for(int t = 0; t < popSize; t++){
    weights[t].resize(numWeights, 0);
    values[t] = 0;
  }

  generateWeights();
}

HillClimbingAgent::~HillClimbingAgent(){
}


void HillClimbingAgent::generateWeights(){

  for(int t = 0; t < popSize; t++){
    for(int i = 0; i < numWeights; i++){
      weights[t][i] = mean[i] + gsl_ran_gaussian(ran, sqrt(variance[i]));
    }
    
    values[t] = 0;
  }

  currentIndex = 0;
  numEvalEpisodes = 0;
}

int HillClimbingAgent::takeAction(const vector<double> &state, const vector<double> &w){

  vector<double> actionValue;

  int s = 0;
  for(int i = 0; i < numActions; i++){

    double val = 0;
    for(int j = 0; j < numFeatures; j++){

      val += state[j] * w[s];
      s++;
    }

    actionValue.push_back(val);
  }
  
  int bestAction = 0;
  double bestVal = actionValue[0];

  int numTies = 0;

  for(int i = 1; i < numActions; i++){
    
    if(fabs(actionValue[i] - bestVal) < EPS){
      numTies++;

      if(gsl_rng_uniform(ran) < (1.0 / (1.0 + numTies))){
	bestVal = actionValue[i];
	bestAction = i;
      }
    }
    else if(actionValue[i] > bestVal){
	bestVal = actionValue[i];
	bestAction = i;
	numTies = 0;
    }
  }

  return bestAction;
}


int HillClimbingAgent::takeAction(const vector<double> &state){

  return takeAction(state, weights[currentIndex]);
}

int HillClimbingAgent::takeBestAction(const vector<double> &state){

  return takeAction(state, bestWeights);
}

void HillClimbingAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

  values[currentIndex] += reward;

  if(terminal){

    numEvalEpisodes++;
    if(numEvalEpisodes == numTotalEvalEpisodes){
      
      if(values[currentIndex] > bestValue){
	
	for(int i = 0; i < numWeights; i++){
	  bestWeights[i] = weights[currentIndex][i];
	}
	bestValue = values[currentIndex];
	//cout << "Best Value: " << bestValue << "\n";
      }

      numEvalEpisodes = 0;
      currentIndex++;

      if(currentIndex == popSize){
	
	bool valueIncreased = false;
	for(int t = 0; t < popSize; t++){
	  if(values[t] > bestValue - EPS){
	    valueIncreased = true;
	    continue;
	  }
	}

	if(!valueIncreased){
	  for(int i = 0; i < numWeights; i++){
	    variance[i] *= epsilon * epsilon;
	  }
	}

	generateWeights();
      }

    }

  }

}

vector<double> HillClimbingAgent::getBestWeights(){

  return bestWeights;
}

void HillClimbingAgent::setMeanAndvariance(const vector<double> &mean, const vector<double> &variance){

  this->mean = mean;
  this->variance = variance;
  
  generateWeights();

  bestWeights = mean;
  bestValue = -INF;

}

void HillClimbingAgent::setBestWeightsAndValue(const vector<double> &w, const double &v){

  bestWeights = w;
  bestValue = v;
}

