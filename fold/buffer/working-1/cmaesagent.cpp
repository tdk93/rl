#include "cmaesagent.h"

CMAESAgent::CMAESAgent(const int &numFeatures, const int &numActions, const int &popSize, const int &evalEpisodes, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;
  
  numWeights = numFeatures * numActions;

  this->popSize = popSize;
  this->numTotalEvalEpisodes = evalEpisodes;

  // Initialise means and variances to 0, 1 respectively.
  for(int i = 0; i < numWeights; i++){

    bestWeights.push_back(0);
  }

  bestValue = -INF;//minus infinity

  double xstart[1000];
  double stddev[1000];
  for(int i = 0; i < numWeights; i++){
    xstart[i] = 0;
    stddev[i] = 1.0;
  }

  arFunvals = cmaes_init(&evo, numWeights, xstart, stddev, randomSeed, popSize, "initials.par");
  cmaes_ReadSignals(&evo, "signals.par");  /* write header and initial values */

  weights.resize(popSize);
  values.resize(popSize);
  for(int t = 0; t < popSize; t++){
    weights[t].resize(numWeights);
  }

  generatePopulation();
}

CMAESAgent::~CMAESAgent(){

  cmaes_exit(&evo); /* release memory */ 
}


void CMAESAgent::generatePopulation(){

  pop = cmaes_SamplePopulation(&evo); /* do not change content of pop */

  for(int t = 0; t < popSize; t++){

    for(int i = 0; i < numWeights; i++){
      weights[t][i] = pop[t][i];
    }

    values[t] = 0;
  }
  
  currentIndex = 0;
  numEvalEpisodes = 0;
}

int CMAESAgent::takeAction(const vector<double> &state, const vector<double> &w){

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


int CMAESAgent::takeAction(const vector<double> &state){

  // currentIndex is the index of the organism currently being evaluated.
  return takeAction(state, weights[currentIndex]);
}

int CMAESAgent::takeBestAction(const vector<double> &state){

  return takeAction(state, bestWeights);
}


void CMAESAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

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
	
	for(int i = 0; i < popSize; i++){
	  arFunvals[i] = -values[i];//Negative -- CMAES code minimises objective function.
	}
	/* update the search distribution used for cmaes_SampleDistribution() */
	cmaes_UpdateDistribution(&evo, arFunvals);  

	generatePopulation();
      }
    }

  }

}


vector<double> CMAESAgent::getBestWeights(){

  return bestWeights;
}

void CMAESAgent::setBestWeightsAndValue(const vector<double> &w, const double &v){

  bestWeights = w;
  bestValue = v;
}

