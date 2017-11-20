#ifndef CMAESAGENT_H
#define CMAESAGENT_H

#include <iostream>
#include <vector>
#include <math.h>

#include "agent.h"
#include "EALib/CMA.h"
#include <EALib/ObjectiveFunction.h>


using namespace std;

class Obj : public ObjectiveFunctionVS<double> {
    
 private:
  vector<double> start;

 public:
  //! Constructor
  Obj(unsigned d, const vector<double> &start);
  
  //! Destructor
  ~Obj();
  
  unsigned int objectives() const;
  void result(double* const& point, std::vector<double>& value);
  bool ProposeStartingPoint(double*& point) const;
  //    bool utopianFitness(std::vector<double>& value) const;
};


class CMAESAgent : public Agent{
  
 private:
  
  int numFeatures;
  int numActions;

  int popSize;
  int numWeights;
  int selectSize;

  int currentIndex;
  int numTotalEvalEpisodes;
  int numEvalEpisodes;

  Obj *obj;

  vector<double> mean;
  //  vector<double> variance;


  vector<double> bestWeights;
  double bestValue;

  // Linear function approximator -- one for each action.
  // Contains a bias term (the last one).
  vector< vector<double> > weights;
  vector<double> values;

  void generatePopulation();
  //  void computeNextMeanAndVariance();

  int takeAction(const vector<double> &state, const vector<double> &w);


 public:

  CMAESAgent(const int &numFeatures, const int &numActions, const int &randomSeed);
  ~CMAESAgent();

  int takeAction(const vector<double> &state);
  int takeBestAction(const vector<double> &state);
  void update(const double &reward, const vector<double> &state, const bool &terminal);

  vector<double> getBestWeights();

  void setMeanAndvariance(const vector<double> &mean, const vector<double> &variance);
  void setBestWeightsAndValue(const vector<double> &w, const double &v);

};

#endif //CMAESAGENT_H

