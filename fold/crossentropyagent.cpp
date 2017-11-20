#include "crossentropyagent.h"

CrossEntropyAgent::CrossEntropyAgent(const int &numFeatures, const int &numActions,  const unsigned long int &totalEpisodes, const int &generations, const int &evalEpisodes, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;
  
  numWeights = numFeatures * numActions;

  // Make the selectsize 15% of popSize.
  this->popSize = (int)((double)totalEpisodes / (evalEpisodes * generations));
  this->selectSize = (this->popSize * 0.15) + 0.5;

  this->numTotalEvalEpisodes = evalEpisodes;

  // Initialise means and variances to 0, 1 respectively.
  for(int i = 0; i < numWeights; i++){

    mean.push_back(0);
    variance.push_back(1.0);

    bestWeights.push_back(0);
  }

  bestValue = -INF;//minus infinity

  for(int t = 0; t < popSize; t++){

    vector<double> w;
    for(int i = 0; i < numWeights; i++){
      w.push_back(0);
    }
    
    weights.push_back(w);
    values.push_back(0);
  }

  generatePopulation();
}

CrossEntropyAgent::~CrossEntropyAgent(){
}


void CrossEntropyAgent::generatePopulation(){

  for(int t = 0; t < popSize; t++){

    for(int i = 0; i < numWeights; i++){
      weights[t][i] = mean[i] + gsl_ran_gaussian(ran, sqrt(variance[i]));
    }

    values[t] = 0;
  }
  
  currentIndex = 0;
  numEvalEpisodes = 0;
}

void CrossEntropyAgent::computeNextMeanAndVariance(){

  // find the numSelect-th order statistic among values.
  vector<double> tempValue = values;

  for(int s = 0; s < selectSize; s++){

    for(int i = 0; i < popSize - s - 1; i++){

      if(tempValue[i] > tempValue[i + 1]){
	double temp = tempValue[i];
	tempValue[i] = tempValue[i + 1];
	tempValue[i + 1] = temp;
      }
    }
  }


  // For all elements with higher values than the chosen order
  // statistic, average and find sample std. deviation to re-seed
  // next population.
  vector<double> sigmaX, sigmaXSquared;
  int n;

  n = 0;
  for(int i = 0; i < numWeights; i++){
    sigmaX.push_back(0);
    sigmaXSquared.push_back(0);
  }

  for(int t = 0; t < popSize; t++){

    if(values[t] >= tempValue[popSize - selectSize]){
      
      n++;
      for(int i = 0; i < numWeights; i++){

	sigmaX[i] += weights[t][i];
	sigmaXSquared[i] += weights[t][i] * weights[t][i];
      }

    }
  }

  for(int i = 0; i < numWeights; i++){
    mean[i] = sigmaX[i] / n;
    variance[i] = (n * sigmaXSquared[i] - (sigmaX[i] * sigmaX[i])) / (n * (n - 1));
  }
}


int CrossEntropyAgent::takeAction(const vector<double> &state, const vector<double> &w){

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


int CrossEntropyAgent::takeAction(const vector<double> &state){

  // currentIndex is the index of the organism currently being evaluated.
  return takeAction(state, weights[currentIndex]);
}

int CrossEntropyAgent::takeBestAction(const vector<double> &state){

  return takeAction(state, bestWeights);
  //return takeAction(state, mean);
}


void CrossEntropyAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

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

      currentIndex++;
      numEvalEpisodes = 0;

      if(currentIndex == popSize){
	
	computeNextMeanAndVariance();
	generatePopulation();
      }
    }

  }

}


vector<double> CrossEntropyAgent::getBestWeights(){

  return bestWeights;
}


void CrossEntropyAgent::setMeanAndVariance(const vector<double> &mean, const vector<double> &variance){

  this->mean = mean;
  this->variance = variance;
  
  generatePopulation();

  bestWeights = mean;
  bestValue = -INF;

  //Hack. Otherwise transfer is lost on CE.
  weights[0] = bestWeights;
  
}

void CrossEntropyAgent::setBestWeightsAndValue(const vector<double> &w, const double &v){

  bestWeights = w;
  bestValue = v;
}

