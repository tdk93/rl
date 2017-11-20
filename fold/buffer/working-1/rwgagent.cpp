#include "rwgagent.h"

RWGAgent::RWGAgent(const int &numFeatures, const int &numActions, const int &evalEpisodes, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;
  
  numWeights = numFeatures * numActions;

  this->numTotalEvalEpisodes = evalEpisodes;

  // Initialise means and variances to 0, 1 respectively.
  for(int i = 0; i < numWeights; i++){

    mean.push_back(0);
    variance.push_back(1.0);

    bestWeights.push_back(0);
  }

  bestValue = -INF;//minus infinity

  for(int i = 0; i < numWeights; i++){
    weights.push_back(0);
  }
  value = 0;

  generateWeights();
}

RWGAgent::~RWGAgent(){
}


void RWGAgent::generateWeights(){

  for(int i = 0; i < numWeights; i++){
    weights[i] = mean[i] + gsl_ran_gaussian(ran, sqrt(variance[i]));
  }

  value = 0;
  
  numEvalEpisodes = 0;
}

int RWGAgent::takeAction(const vector<double> &state, const vector<double> &w){

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


int RWGAgent::takeAction(const vector<double> &state){

  return takeAction(state, weights);
}

int RWGAgent::takeBestAction(const vector<double> &state){

  return takeAction(state, bestWeights);
}

void RWGAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

  value += reward;

  if(terminal){

    numEvalEpisodes++;
    if(numEvalEpisodes == numTotalEvalEpisodes){
      
      if(value > bestValue){
	
	for(int i = 0; i < numWeights; i++){
	  bestWeights[i] = weights[i];
	}
	bestValue = value;
	//cout << "Best Value: " << bestValue << "\n";
      }

      numEvalEpisodes = 0;

      generateWeights();
    }

  }

}

vector<double> RWGAgent::getBestWeights(){

  return bestWeights;
}

void RWGAgent::setMeanAndvariance(const vector<double> &mean, const vector<double> &variance){

  this->mean = mean;
  this->variance = variance;
  
  generateWeights();

  bestWeights = mean;
  bestValue = -INF;

  //Hack. Otherwise transfer is lost.
  weights = bestWeights;
  
}

void RWGAgent::setBestWeightsAndValue(const vector<double> &w, const double &v){

  bestWeights = w;
  bestValue = v;
}

