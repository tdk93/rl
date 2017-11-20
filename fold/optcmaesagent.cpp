#include "optcmaesagent.h"

OptCMAESAgent::OptCMAESAgent(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const int &generations, const int &evalEpisodes, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;
  
  numWeights = numFeatures * numActions;

  this->popSize = (int)((double)totalEpisodes / (evalEpisodes * generations));
  this->numTotalEvalEpisodes = evalEpisodes;

  //  cout << "popSize: " << popSize << "\n";

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

  //Selectsize is half of popSize in the default parameter set used by CMA-ES (cmaes.cpp).
  selectSize = popSize / 2;
  arFunvals = cmaes_init_mu(&evo, numWeights, xstart, stddev, randomSeed, popSize, selectSize, "initials.par");
  
  cmaes_ReadSignals(&evo, "signals.par");  /* write header and initial values */

  weights.clear();
  values.clear();
  meanValues.clear();

  weights.resize(popSize);
  values.resize(popSize);
  for(int t = 0; t < popSize; t++){
    weights[t].resize(numWeights);
    values[t].resize(0);
  }
  meanValues.resize(popSize, 0);

  generatePopulation();
}

OptCMAESAgent::OptCMAESAgent(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const int &generations, const int &evalEpisodes, const int &randomSeed, const vector<double> &startMean, const vector<double> &startVariance) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;
  
  numWeights = numFeatures * numActions;

  this->popSize = (int)((double)totalEpisodes / (evalEpisodes * generations));
  this->numTotalEvalEpisodes = evalEpisodes;

  //  cout << "popSize: " << popSize << "\n";

  // Initialise means and variances to 0, 1 respectively.
  for(int i = 0; i < numWeights; i++){

    bestWeights.push_back(0);
  }

  bestValue = -INF;//minus infinity

  double xstart[1000];
  double stddev[1000];
  for(int i = 0; i < numWeights; i++){
    xstart[i] = startMean[i];
    stddev[i] = sqrt(startVariance[i]);
  }

  //Selectsize is half of popSize in the default parameter set used by CMA-ES (cmaes.cpp).
  selectSize = popSize / 2;
  arFunvals = cmaes_init_mu(&evo, numWeights, xstart, stddev, randomSeed, popSize, selectSize, "initials.par");

  cmaes_ReadSignals(&evo, "signals.par");  /* write header and initial values */

  weights.clear();
  values.clear();
  meanValues.clear();

  weights.resize(popSize);
  values.resize(popSize);
  for(int t = 0; t < popSize; t++){
    weights[t].resize(numWeights);
    values[t].resize(0);
  }
  meanValues.resize(popSize, 0);

  generatePopulation();
}

OptCMAESAgent::~OptCMAESAgent(){

  cmaes_exit(&evo); /* release memory */ 
}


void OptCMAESAgent::generatePopulation(){

  pop = cmaes_SamplePopulation(&evo); /* do not change content of pop */

  const double *xMean = cmaes_GetPtr(&evo, "xmean");
  mean.clear();
  mean.resize(numWeights);
  for(int i = 0; i < numWeights; i++){
    mean[i] = xMean[i];
  }

  for(int t = 0; t < popSize; t++){

    for(int i = 0; i < numWeights; i++){
      weights[t][i] = pop[t][i];
    }

    values[t].clear();
    values[t].resize(0);
    meanValues[t] = 0;
  }
  
  currentEpisodeReward = 0;
  eliminated.clear();
  eliminated.resize(popSize, false);
  numEliminated = 0;

  currentIndex = getIndexToSample();
  numEvalEpisodes = 0;
}

int OptCMAESAgent::takeAction(const vector<double> &state, const vector<double> &w){

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


int OptCMAESAgent::takeAction(const vector<double> &state){

  // currentIndex is the index of the organism currently being evaluated.
  return takeAction(state, weights[currentIndex]);
}

int OptCMAESAgent::takeBestAction(const vector<double> &state){

  return takeAction(state, bestWeights);
  //return takeAction(state, mean);
}

void OptCMAESAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

  currentEpisodeReward += reward;

  if(terminal){

    values[currentIndex].push_back(currentEpisodeReward);
    meanValues[currentIndex] *= values[currentIndex].size() - 1;
    meanValues[currentIndex] += currentEpisodeReward;
    meanValues[currentIndex] /= values[currentIndex].size();

    currentEpisodeReward = 0;
    numEvalEpisodes++;

    if(numEvalEpisodes == numTotalEvalEpisodes * popSize){

      for(int p = 0; p < popSize; p++){
	if(meanValues[p] > bestValue){
	  for(int i = 0; i < numWeights; i++){
	    bestWeights[i] = weights[p][i];
	  }
	  bestValue = meanValues[p];
	  //cout << "Best Value: " << bestValue << "\n";
	}
      }

      numEvalEpisodes = 0;

      for(int i = 0; i < popSize; i++){
	arFunvals[i] = -meanValues[i];//Negative -- CMAES code minimises objective function.
      }
      // update the search distribution used for cmaes_SampleDistribution()
      cmaes_UpdateDistribution(&evo, arFunvals);  
      generatePopulation();
    }

    currentIndex = getIndexToSample();
  }

}

int OptCMAESAgent::getIndexToSample(){

  //If some individual has never been sampled, sample one.
  for(int p = 0; p < popSize ; p++){
    if(values[p].size() == 0){
      return p;
    }
  }

  // Formula from paper for how many samples must have been gathered at the end of
  // phase (numEliminated + 1).
  double nr1 = (popSize * numTotalEvalEpisodes) - popSize;
  double dr1 = selectSize / 2.0;
  for(int t = 2; t <= popSize - selectSize + 1; t++){
    dr1 += 1.0 / t;
  }
  double dr2 =  popSize - selectSize + 2 - (numEliminated + 1);
  int b = (int)((nr1 / (dr1 * dr2)) + 0.5);//ceil


  // Among arms that have not been eliminated, what is the minimum number of samples?
  int minSamplesAmongSurviving = -1;
  int minSampledIndex = -1;

  for(int p = 0; p < popSize; p++){

    if(!eliminated[p]){

      if(minSampledIndex == -1 || (values[p].size() < minSamplesAmongSurviving)){
	minSamplesAmongSurviving = values[p].size();
	minSampledIndex = p;
      }

    }
  }

  //Eliminate one.
  if(numEliminated < (popSize - selectSize) && minSamplesAmongSurviving >= b){
    
    int lowestIndex = -1;
    for(int p = 0; p < popSize; p++){
      if(!eliminated[p]){
	if(lowestIndex == -1 || meanValues[p] < meanValues[lowestIndex]){
	  lowestIndex = p;
	}
      }
    }

    eliminated[lowestIndex] = true;
    numEliminated++;
  }
  
  // Recalculate after the elimination. This repetition can be avoided to make the code more efficient.
  minSamplesAmongSurviving = -1;
  minSampledIndex = -1;
  for(int p = 0; p < popSize; p++){

    if(!eliminated[p]){
      if(minSampledIndex == -1 || (values[p].size() < minSamplesAmongSurviving)){
	minSamplesAmongSurviving = values[p].size();
	minSampledIndex = p;
      }
    }
  }

  return minSampledIndex;
}

vector<double> OptCMAESAgent::getBestWeights(){

  return bestWeights;
}

void OptCMAESAgent::setBestWeightsAndValue(const vector<double> &w, const double &v){

  bestWeights = w;
  bestValue = v;
}

