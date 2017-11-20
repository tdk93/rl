#include "geneticalgorithmagent.h"

GeneticAlgorithmAgent::GeneticAlgorithmAgent(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const int &generations, const int &evalEpisodes, const int &randomSeed) : Agent(numFeatures, numActions, randomSeed){

  this->numFeatures = numFeatures;
  this->numActions = numActions;
  
  numWeights = numFeatures * numActions;

  // Make the selectsize 15% of popSize.
  this->popSize = (int)((double)totalEpisodes / (evalEpisodes * generations));
  this->selectSize = (this->popSize * 0.15) + 0.5;

  if(selectSize < 1){
    selectSize = 1;
  }

  this->numTotalEvalEpisodes = evalEpisodes;

  numBits = 32;

  min.resize(numWeights, -1.0);
  max.resize(numWeights, 1.0);

  for(int i = 0; i < numWeights; i++){
    bestWeights.push_back(0);
  }
  bestValue = -INF;//minus infinity

  mutationProbability = 0.05;
  mutationBitFlipProbability = 0.1;

  initialisePopulation();
}

GeneticAlgorithmAgent::~GeneticAlgorithmAgent(){
}


/*
  binaryToGray and grayToBinary pseudocode picked up from:
  "http://mathworld.wolfram.com/GrayCode.html".
  "binary" is a bit repr. as an array of 0's and 1's, with
  usual interpretation.
 */
vector<int> GeneticAlgorithmAgent::binaryToGray(const vector<int> &bin){

  vector<int> gray = bin;

  for(int i = numBits - 1; i > 0; i--){

    if(bin[i - 1]){
      gray[i] = 1 - gray[i];
    }

  }

  return gray;
}

vector<int> GeneticAlgorithmAgent::grayToBinary(const vector<int> &gray){

  vector<int> bin = gray;
  vector<int> sigma;
  sigma.resize(numBits);
 
  sigma[0] = 0;

  for(int i = 1; i < numBits; i++){
    sigma[i] = (gray[i - 1] + sigma[i - 1]) % 2;
  }
  
  for(int i = 0; i < numBits; i++){

    if(sigma[i]){
      bin[i] = 1 - bin[i];
    }

  }

  return bin;
}


// Binary: MSB leftmost.
unsigned long int GeneticAlgorithmAgent::binaryToInteger(const vector<int> &binary){
  
  unsigned long int sum = 0;
  for(int bit = 0; bit < numBits; bit++){
    sum *= 2;
    sum += binary[bit];
  }

  return sum;
}

// Assumes integer lies in [0, 2^numBits - 1].
vector<int> GeneticAlgorithmAgent::integerToBinary(const unsigned long int &integer){

  vector<int> bin;
  bin.resize(numBits, 0);

  unsigned long int intCopy = integer;

  for(int i = numBits - 1; i >= 0; i--){
    
    bin[i] = intCopy % 2;
    intCopy = intCopy / 2;
  }

  return bin;
}


/*
  The range [min, max] is divided into (2^numbits - 1) sections. An integer 
  in [0..(2^numBits - 1)] is mapped to each section.
 */
unsigned long int GeneticAlgorithmAgent::doubleToInteger(const double &d, const double &min, const double &max){

  double dMap = (d - min) * (pow(2.0, numBits) - 1) / (max - min);
  return (unsigned long int)(dMap + 0.5);
}

double GeneticAlgorithmAgent::integerToDouble(const unsigned long int &u, const double &min, const double &max){

  return (((double)(u) * (max - min) / (pow(2.0, numBits) - 1)) + min);
}


vector<int> GeneticAlgorithmAgent::doubleToGray(const double &d, const double &min, const double &max){

  return binaryToGray(integerToBinary(doubleToInteger(d, min, max)));
}


double GeneticAlgorithmAgent::grayToDouble(const vector<int> &gray, const double &min, const double &max){

  return integerToDouble(binaryToInteger(grayToBinary(gray)), min, max);
}

void GeneticAlgorithmAgent::initialisePopulation(){

  weights.resize(popSize);
  values.resize(popSize);

  for(int p = 0; p < popSize; p++){

    weights[p].resize(numWeights);

    for(int d = 0; d < numWeights; d++){

      weights[p][d] = (gsl_rng_uniform(ran) * (max[d] - min[d])) + min[d];
    }

    values[p] = 0;
  }

  currentIndex = 0;
  numEvalEpisodes = 0;

}

vector<double> GeneticAlgorithmAgent::mutate(const vector<double> &p){

  vector<double> q = p;
  for(int d = 0; d < numWeights; d++){

    vector<int> gray = doubleToGray(q[d], min[d], max[d]);

    for(int b = 0; b < numBits; b++){
      if(gsl_rng_uniform(ran) < mutationBitFlipProbability){
	gray[b] = 1 - gray[b];
      }
    }

    q[d] = grayToDouble(gray, min[d], max[d]);
  }

  return q;
}


void GeneticAlgorithmAgent::crossover(const vector<double> &p1, const vector<double> &p2, vector<double> &c1, vector<double> &c2){

  c1.resize(numWeights);
  c2.resize(numWeights);

  for(int d = 0; d < numWeights; d++){

    vector<int> gp1 = doubleToGray(p1[d], min[d], max[d]);
    vector<int> gp2 = doubleToGray(p2[d], min[d], max[d]);

    vector<int> gc1 = gp1;
    vector<int> gc2 = gp2;

    int crossoverIndex = (int)(gsl_rng_uniform(ran) * numBits) % numBits;
    for(int b = 0; b <= crossoverIndex; b++){
      gc1[b] = gp2[b];
      gc2[b] = gp1[b];
    }

    c1[d] = grayToDouble(gc1, min[d], max[d]);
    c2[d] = grayToDouble(gc2, min[d], max[d]);
  }

}

void GeneticAlgorithmAgent::iterate(){

  // Sort points in decreasing order of values.
  for(int sweep = 0; sweep < popSize - 1; sweep++){
    for(int p = 0; p < popSize - sweep - 1; p++){

      if(values[p] < values[p + 1]){

	double tempValue = values[p];
	values[p] = values[p + 1];
	values[p + 1] = tempValue;

	vector<double> tempPoint = weights[p];
	weights[p] = weights[p + 1];
	weights[p + 1] = tempPoint;
      }
    }
  }

  // If bestValue has been improved, update bestPoint.
  if(values[0] > bestValue){
    bestWeights = weights[0];
    bestValue = values[0];
  }


  vector< vector<double> > children;
  int ctr = 0;

  while(ctr < popSize){

    int parent1Index = (int)(gsl_rng_uniform(ran) * selectSize) % selectSize;
    int parent2Index = (int)(gsl_rng_uniform(ran) * selectSize) % selectSize;

    vector<double> child1, child2;

    crossover(weights[parent1Index], weights[parent2Index], child1, child2);

    children.push_back(child1);
    children.push_back(child2);
    ctr += 2;
  }

  for(int p = 0; p < popSize; p++){

    if(gsl_rng_uniform(ran) < mutationProbability){
      weights[p] = mutate(children[p]);
    }
    else{
      weights[p] = children[p];
    }
    
    values[p] = 0;
  }

  currentIndex = 0;
  numEvalEpisodes = 0;

}


int GeneticAlgorithmAgent::takeAction(const vector<double> &state, const vector<double> &w){

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


int GeneticAlgorithmAgent::takeAction(const vector<double> &state){

  // currentIndex is the index of the organism currently being evaluated.
  return takeAction(state, weights[currentIndex]);
}

int GeneticAlgorithmAgent::takeBestAction(const vector<double> &state){

  return takeAction(state, bestWeights);
}


void GeneticAlgorithmAgent::update(const double &reward, const vector<double> &state, const bool &terminal){

  values[currentIndex] += reward;

  if(terminal){

    numEvalEpisodes++;
    if(numEvalEpisodes == numTotalEvalEpisodes){
      
      if(values[currentIndex] > bestValue){
	
	for(int i = 0; i < numWeights; i++){
	  bestWeights[i] = weights[currentIndex][i];
	}
	bestValue = values[currentIndex];
	//	cout << "Best Value: " << bestValue << "\n";
      }

      currentIndex++;
      numEvalEpisodes = 0;

      if(currentIndex == popSize){
	
	iterate();
      }
    }

  }

}


vector<double> GeneticAlgorithmAgent::getBestWeights(){

  return bestWeights;
}


void GeneticAlgorithmAgent::setBestWeightsAndValue(const vector<double> &w, const double &v){

  bestWeights = w;
  bestValue = v;
}

