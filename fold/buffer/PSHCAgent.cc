#include "PSHCAgent.h"

PSHCAgent::PSHCAgent(int numFeatures, int numActions) : Agent(numFeatures, numActions){

  this->numFeatures = numFeatures;
  this->numActions = numActions;

  numWorseEvals = 0;
  maxWorseEvals = 100;
  numTotalEvalEpisodes = 25;
  numEvalEpisodes = 0;

  maxWeightDelta = 0.10;
  currentWeightDelta = maxWeightDelta;
  deltaDecayFraction = 0.9;

  srand48(time(0));

  for(int a = 0; a < numActions; a++){
    for(int f = 0; f <= numFeatures; f++){
      currentWeights[a][f] = 0;
      nextWeights[a][f] = 0;
    }
  }
  currentValue = -1000000.0;//minus infinity.
  nextValue = 0;//minus infinity.

}

PSHCAgent::~PSHCAgent(){
}

void PSHCAgent::generateNext(){

  for(int a = 0; a < numActions; a++){
    for(int f = 0; f <= numFeatures; f++){
      double addend = currentWeightDelta;
      if(drand48() < 0.5){
	addend = -addend;
      }
      nextWeights[a][f] = currentWeights[a][f] + addend;
    }
  }
  
  nextValue = 0;//minus infinity
}

int PSHCAgent::takeAction(double state[]){
  
  double bestVal;
  int bestAction;

  int numTies = 0;
  double EPS = 1.0e-6;

  for(int a = 0; a < numActions; a++){

    double value = 0;
    for(int f = 0; f < numFeatures; f++){
      value += state[f] * nextWeights[a][f];
    }
    value += nextWeights[a][numFeatures];

    if(a == 0){
      bestVal = value;
      bestAction = a;
    }
    else if(fabs(bestVal - value) < EPS){
      numTies++;
      if(drand48() < (1.0 / (1.0 + numTies))){
	bestVal = value;
	bestAction = a;
      }
    }
    else if(value > bestVal){
      numTies = 0;
      bestVal = value;
      bestAction = a;
    }
  }

  return bestAction;
}


int PSHCAgent::takeBestAction(double state[]){
  
  double bestVal;
  int bestAction;

  int numTies = 0;
  double EPS = 1.0e-6;

  for(int a = 0; a < numActions; a++){

    double value = 0;
    for(int f = 0; f < numFeatures; f++){
      value += state[f] * currentWeights[a][f];
    }
    value += currentWeights[a][numFeatures];

    if(a == 0){
      bestVal = value;
      bestAction = a;
    }
    else if(fabs(bestVal - value) < EPS){
      numTies++;
      if(drand48() < (1.0 / (1.0 + numTies))){
	bestVal = value;
	bestAction = a;
      }
    }
    else if(value > bestVal){
      numTies = 0;
      bestVal = value;
      bestAction = a;
    }
  }

  return bestAction;
}

void PSHCAgent::update(double reward, double state[], bool terminal){

  nextValue += reward;

  if(terminal){

    numEvalEpisodes++;
    
    if(numEvalEpisodes == numTotalEvalEpisodes){
      numEvalEpisodes = 0;

      double EPS = 1.0e-6;

      if((nextValue > currentValue) || ((fabs(nextValue - currentValue) < EPS) && drand48() < 0.5)){
	
	currentValue = nextValue;
	for(int a = 0; a < numActions; a++){
	  for(int i = 0; i <= numFeatures; i++){
	    currentWeights[a][i] = nextWeights[a][i];
	  }
	}

	//	currentWeightDelta = maxWeightDelta;
      }
      else{
	numWorseEvals++;

	if(numWorseEvals == maxWorseEvals){
	  numWorseEvals = 0;
	  
	  currentWeightDelta *= deltaDecayFraction;
	  cout << "Decaying weightDelta to: " << currentWeightDelta << "\n";
	}
      }

      generateNext();
    }
  }
}

