#include "cmaesagent.h"

CMAESAgent::CMAESAgent(const int &numFeatures, const int &numActions, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;
  
  //  this->popSize = 100;
  this->popSize = 50;
    numWeights = (numFeatures + 1) * numActions;
  //  selectSize = 10;
  numTotalEvalEpisodes = 4000;

  double mean[numWeights];
  double stdDev[numWeights];
  for(int w = 0; w < numWeights; w++){
    mean[w] = 0;
    stdDev[w] = 10.1;

    bestWeights.push_back(0);
  }
  bestValue = -INF;//minus infinity

  
  long seed = gsl_rng_get(ran);

  //arFunvals = cmaes_init(&evo, numFeatures, mean, stdDev, seed, popSize, "initials.par");
  cmaes_init(&evo, numWeights, mean, stdDev, seed, popSize, "initials.par");
  
  printf("%s\n", cmaes_SayHello(&evo));
  //  int g;
  //  cin >> g;

  //  cmaes_ReadSignals(&evo, "signals.par");  /* write header and initial values */

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

CMAESAgent::~CMAESAgent(){

  cmaes_exit(&evo);
}


void CMAESAgent::generatePopulation(){

  double *const*pop;
  pop = cmaes_SamplePopulation(&evo);

  for(int t = 0; t < popSize; t++){

    for(int w = 0; w < numWeights; w++){
      weights[t][w] = pop[t][w];
    }

    values[t] = 0;
  }
  
  currentIndex = 0;
  numEvalEpisodes = 0;
}

/*
void CMAESAgent::computeNextMeanAndVariance(){

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
*/


int CMAESAgent::takeAction(const vector<double> &state, const vector<double> &w){

  vector<double> actionValue;

  int s = 0;
  for(int i = 0; i < numActions; i++){

    double val = 0;
    for(int j = 0; j < numFeatures; j++){

      val += state[j] * w[s];
      s++;
    }

    val += w[s];
    s++;

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
	cout << "Best Value: " << bestValue << "\n";
      }

      currentIndex++;
      numEvalEpisodes = 0;

      if(currentIndex == popSize){
	
	/////////	computeNextMeanAndVariance();

	double arFunvals[popSize];
	for(int t = 0; t < popSize; t++){
	  arFunvals[t] = values[t];
	}
	
	/* update the search distribution used for cmaes_SampleDistribution() */
	cmaes_UpdateDistribution(&evo, arFunvals);  
  printf("%s\n", cmaes_SayHello(&evo));
  
	//cmaes_ReadSignals(&evo, "signals.par");   
	//	fflush(stdout); /* useful in MinGW */


	generatePopulation();

      }
    }

  }

}


vector<double> CMAESAgent::getBestWeights(){

  return bestWeights;
}


void CMAESAgent::setMeanAndvariance(const vector<double> &mean, const vector<double> &variance){

  //  this->mean = mean;
  //  this->variance = variance;

  double m[numWeights];
  double s[numWeights];
  for(int w = 0; w < numWeights; w++){
    m[w] = mean[w];
    s[w] = sqrt(variance[w]);
  }
  
  long seed = gsl_rng_get(ran);
  //  arFunvals = cmaes_init(&evo, numFeatures, mean, stdDev, seed, popSize, "initials.par");
  cmaes_init(&evo, numWeights, m, s, seed, popSize, "initials.par");
  printf("%s\n", cmaes_SayHello(&evo));
  //  cmaes_ReadSignals(&evo, "signals.par");  /* write header and initial values */

 
  generatePopulation();

  bestWeights = mean;
  bestValue = -INF;
}

void CMAESAgent::setBestWeightsAndValue(const vector<double> &w, const double &v){

  bestWeights = w;
  bestValue = v;
}

