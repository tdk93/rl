#ifndef GENETICALGORITHMAGENT_H
#define GENETICALGORITHMAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"

using namespace std;

/*
  Implement genetic algorithm method. Input parameters: Population size,
  numEvalEpisodes. Mutation probabilities are manually tuned.
 */


class GeneticAlgorithmAgent : public Agent{

 private:

  int numFeatures;
  int numActions;

  int numWeights;

  int popSize;
  int selectSize;
  int currentIndex;

  int numTotalEvalEpisodes;
  int numEvalEpisodes;

  // min and max are bounds in each dimension.
  vector<double> min;
  vector<double> max;

  // On each iteration, with probability mutationProbability
  // mutate an individual. If mutating an individual, flip
  // each of its bits (in gray code repr.) with probability
  // mutationBitFlipProbability.
  double mutationProbability;
  double mutationBitFlipProbability;

  // Number of bits in the gray code representation.
  int numBits;

  vector<double> bestWeights;
  double bestValue;

  // Linear function approximator -- one for each action.
  vector< vector<double> > weights;
  vector<double> values;

  // Utulities for representation conversion.
  vector<int> binaryToGray(const vector<int> &bin);
  vector<int> grayToBinary(const vector<int> &gray);
  unsigned long int binaryToInteger(const vector<int> &binary);
  vector<int> integerToBinary(const unsigned long int &integer);
  unsigned long int doubleToInteger(const double &d, const double &min, const double &max);
  double integerToDouble(const unsigned long int &u, const double &min, const double &max);

  // Main conversion functions used, which cascade primitives.
  vector<int> doubleToGray(const double &d, const double &min, const double &max);
  double grayToDouble(const vector<int> &gray, const double &min, const double &max);

  void initialisePopulation();
  void iterate();

  vector<double> mutate(const vector<double> &p);
  void crossover(const vector<double> &p1, const vector<double> &p2, vector<double> &c1, vector<double> &c2);

  int takeAction(const vector<double> &state, const vector<double> &w);

 public:

  GeneticAlgorithmAgent(const int &numFeatures, const int &numActions, const unsigned long int &totalEpisodes, const int &generations, const int &evalEpisodes, const int &randomSeed);
  ~GeneticAlgorithmAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

  vector<double> getBestWeights();

  void setBestWeightsAndValue(const vector<double> &w, const double &v);

};

#endif //GENETICALGORITHMAGENT_H

