#include "PSHCAgentSteep.h"

PSHCAgentSteep::PSHCAgentSteep(int numFeatures, int numActions) : Agent(numFeatures, numActions){

  this->numFeatures = numFeatures;
  this->numActions = numActions;

  popSize = 50;
  numTotalEvalEpisodes = 25;
  numEvalEpisodes = 0;
  weightDelta = 0.05;
  deltaDecayFraction = 0.9;

  srand48(time(0));

  for(int a = 0; a < numActions; a++){
    for(int f = 0; f <= numFeatures; f++){
      currentWeights[a][f] = 0;
    }
  }
  currentValue = -1000000.0;//minus infinity.

  generatePopulation();
  numEvaluated = 0;
}

PSHCAgentSteep::~PSHCAgentSteep(){
}

void PSHCAgentSteep::generatePopulation(){


  for(int i = 0; i < popSize; i++){
    for(int a = 0; a < numActions; a++){
      for(int f = 0; f <= numFeatures; f++){
	double addend = weightDelta;
	if(drand48() < 0.5){
	  addend = -addend;
	}
	weights[i][a][f] = currentWeights[a][f] + addend;
	values[i] = 0;
      }
    }
  }

}

int PSHCAgentSteep::takeAction(double state[]){
  
  double bestVal;
  int bestAction;

  int numTies = 0;
  double EPS = 1.0e-6;

  for(int a = 0; a < numActions; a++){

    double value = 0;
    for(int f = 0; f < numFeatures; f++){
      value += state[f] * weights[numEvaluated][a][f];
    }
    value += weights[numEvaluated][a][numFeatures];

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


int PSHCAgentSteep::takeBestAction(double state[]){
  
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


void PSHCAgentSteep::update(double reward, double state[], bool terminal){

  values[numEvaluated] += reward;

  //  cout << "AAAAAAAAA: " << reward << ", " << terminal << "\n";

  if(terminal){

    //    cout << numEvaluated << ", " << numEvalEpisodes << "\n";

    numEvalEpisodes++;
    
    if(numEvalEpisodes == numTotalEvalEpisodes){
      numEvaluated++;
      numEvalEpisodes = 0;
    }
    
    if(numEvaluated == popSize){
      numEvaluated = 0;
      
      double bestVal;
      int bestIndex;
      int numTies = 0;
      double EPS = 1.0e-6;
      
      cout << "Iterating...\n";
      cout << "currentValue: " << currentValue << "\n";
      
      for(int i = 0; i < popSize; i++){
	
	cout << "value[" << i << "]: " << values[i] << "\n";
	
	if(i == 0){
	  bestVal = values[i];
	  bestIndex = i;
	}
	else if(fabs(bestVal - values[i]) < EPS){
	  numTies++;
	  if(drand48() < (1.0 / (1.0 + numTies))){
	    bestVal = values[i];
	    bestIndex = i;
	  }
	}
	else if(values[i] > bestVal){
	  numTies = 0;
	  
	  bestVal = values[i];
	  bestIndex = i;
	}
      }
      
      if(bestVal > currentValue || (fabs(bestVal - currentValue) < EPS && drand48() < 0.5)){
	
	for(int a = 0; a < numActions; a++){
	  for(int i = 0; i <= numFeatures; i++){
	    currentWeights[a][i] = weights[bestIndex][a][i];
	  }
	}
	currentValue = bestVal;
      }
      else{
	
	weightDelta *= deltaDecayFraction;
	cout << "Decaying weightDelta to: " << weightDelta << "\n";
      }
      
      cout << "Updated current value: " << currentValue << "\n";
      
      generatePopulation();
    }
    
  }

}

