#include "cmaesagent.h"



Obj::Obj(unsigned d, const vector<double> &start) : ObjectiveFunctionVS<double>(d, NULL){
  m_name = "General Objective Function";
  this->start = start;
}

Obj::~Obj(){
}


unsigned int Obj::objectives() const{
  return 1;
}


void Obj::result(double* const& point, std::vector<double>& value){

  /*	unsigned i;
	double sum = 0.;
	for (i = 0; i < m_dimension; i++) sum += point[i] * point[i];
	value.resize(1);
	value[0] = sum;
	m_timesCalled++;
  */

  //evaluate fitness of point here.


  value.resize(1);
  value[0] = 1.0;
}

bool Obj::ProposeStartingPoint(double*& point) const{

  for(int i = 0; i < m_dimension; i++){
    point[i] = start[i];
  }

  return true;
}

/*
bool CMAESAgent::Obj::utopianFitness(std::vector<double>& fitness) const{

  fitness.resize(1, false);
  fitness[0] = 0.0;
  return true;
}
*/







CMAESAgent::CMAESAgent(const int &numFeatures, const int &numActions, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;
  
  this->popSize = 100;
  numWeights = (numFeatures + 1) * numActions;
  selectSize = 10;
  numTotalEvalEpisodes = 2000;

  vector<double> mean;
  vector<double> stdDev;
  mean.resize(numWeights, 0);
  stdDev.resize(numWeights, 1.0);
  bestWeights.resize(numWeights, 0);

  //  for(int w = 0; w < numWeights; w++){
  //    mean[w] = 0;
  //    stdDev[w] = 1.0;

  //    bestWeights.push_back(0);
  //  }
  bestValue = -INF;//minus infinity

  obj = new Obj(numWeights, mean);
  
  //  const unsigned Iterations     = 600;
  const double   GlobalStepInit = 1.;
  
  Array<double> range;
  for(int w = 0; w < numWeights; w++){
    double min = -10;
    double max = -10;
    double r = min + (gsl_rng_uniform(ran) * (max - min));
    range.push_back(r);
  }
  
  // search algorithm
  CMASearch cma;
  cma.init(f, range, GlobalStepInit);


  /*


  long seed = gsl_rng_get(ran);

  //arFunvals = cmaes_init(&evo, numFeatures, mean, stdDev, seed, popSize, "initials.par");
  //  cmaes_init(&evo, numWeights, mean, stdDev, seed, popSize, "initials.par");
  
  //  printf("%s\n", cmaes_SayHello(&evo));
  //  int g;
  //  cin >> g;

  for(int t = 0; t < popSize; t++){

    vector<double> w;
    for(int i = 0; i < numWeights; i++){
      w.push_back(0);
    }
    
    weights.push_back(w);
    values.push_back(0);
  }

  generatePopulation();
  */
}

CMAESAgent::~CMAESAgent(){

  delete obj;

  //  cmaes_exit(&evo);
}


void CMAESAgent::generatePopulation(){

    double *const*pop;
  //  pop = cmaes_SamplePopulation(&evo);

  for(int t = 0; t < popSize; t++){

    for(int w = 0; w < numWeights; w++){
      weights[t][w] = pop[t][w];
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
	//	cmaes_UpdateDistribution(&evo, arFunvals);  
	//  printf("%s\n", cmaes_SayHello(&evo));
  
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
  //  cmaes_init(&evo, numWeights, m, s, seed, popSize, "initials.par");
  //  printf("%s\n", cmaes_SayHello(&evo));
  //  cmaes_ReadSignals(&evo, "signals.par");  /* write header and initial values */

 
  generatePopulation();

  bestWeights = mean;
  bestValue = -INF;
}

void CMAESAgent::setBestWeightsAndValue(const vector<double> &w, const double &v){

  bestWeights = w;
  bestValue = v;
}

