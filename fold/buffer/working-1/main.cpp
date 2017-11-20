#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <math.h>

#include "gsl/gsl_rng.h"
#include "headers.h"
#include "mdp.h"

#include "randomagent.h"
#include "sarsalambdaagent.h"
#include "consarsalambdaagent.h"
#include "hlsarsalambdaagent.h"
#include "expsarsalambdaagent.h"
#include "conexpsarsalambdaagent.h"
#include "greedygqlambdaagent.h"
#include "qlearningagent.h"
#include "crossentropyagent.h"
#include "cmaesagent.h"
#include "geneticalgorithmagent.h"
#include "hillclimbingagent.h"
#include "rwgagent.h"
//#include "transferagent.h"

using namespace std;


void options(){

  cout << "Usage:\n";
  cout << "mdp-evaluate\n"; 
  cout << "\t[--s s]\n";
  cout << "\t[--p p]\n";
  cout << "\t[--chi chi]\n";
  cout << "\t[--w w]\n";
  cout << "\t[--sigma sigma]\n"; 
  cout << "\t[--method random | sarsa_lambda_alphainit_epsinit | expsarsa_lambda_alphainit_epsinit | greedygq_lambda_alphainit_epsinit | qlearning_alphainit_epsinit | ce_popsize_evalepisodes | cmaes_popsize_evalepisodes | ga_popsize_evalepisodes | transfer_lambda_evalepisodes | rwg_evalepisodes]\n";
  cout << "\t[--randomSeed randomSeed]\n";
  cout << "\t[--outFile outFile]\n";
}


//  Read command line arguments, and set the ones that are passed (the others remain default.)
bool setRunParameters(int argc, char *argv[], int &s, double &p, double &chi, int &w, double &sigma, string &method, int &randomSeed, string &outFileName){

  int ctr = 1;
  while(ctr < argc){

    cout << string(argv[ctr]) << "\n";

    if(string(argv[ctr]) == "--help"){
      return false;//This should print options and exit.
    }
    else if(string(argv[ctr]) == "--s"){
      if(ctr == (argc - 1)){
	return false;
      }
      s = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--p"){
      if(ctr == (argc - 1)){
	return false;
      }
      p = atof(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--chi"){
      if(ctr == (argc - 1)){
	return false;
      }
      chi = atof(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--w"){
      if(ctr == (argc - 1)){
	return false;
      }
      w = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--sigma"){
      if(ctr == (argc - 1)){
	return false;
      }
      sigma = atof(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--method"){
      if(ctr == (argc - 1)){
	return false;
      }
      method = argv[ctr + 1];
      ctr++;
    }
    else if(string(argv[ctr]) == "--randomSeed"){
      if(ctr == (argc - 1)){
	return false;
      }
      randomSeed = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--outFile"){
      if(ctr == (argc - 1)){
	return false;
      }
      outFileName = string(argv[ctr + 1]);
      ctr++;
    }
    else{
      return false;
    }

    ctr++;
  }

  return true;
}


// Run agent on m for numEpisodes, and compute average reward. No discounting.
double evaluate(MDP *m, Agent *agent, const int &numEpisodes){

  vector<double> features;
  
  double totalReward = 0;
  
  m->reset();

  for(int i = 0; i < numEpisodes; i++){

    bool term;
    do{

      features = m->getFeatures();
      int action = agent->takeBestAction(features);
      term = m->takeAction(action);
      double r = m->getLastReward();
      totalReward += r;
    }
    while(term == false);

    //cout << "Episode " << i << " Total Reward: " << totalReward << "\n";
  }

  double averageReward = totalReward / numEpisodes;
  
  return averageReward;
}


void test(MDP *m){

  char ac;
  do{
    
    m->display(DISPLAY_FEATURES);
    m->display(DISPLAY_TERMINAL);
    m->display(DISPLAY_REWARDS);
    m->display(DISPLAY_MAXIMAL_VALUES);
    m->display(DISPLAY_MINIMAL_VALUES);
    m->display(DISPLAY_MAXIMAL_ACTIONS);
    m->display(DISPLAY_MINIMAL_ACTIONS);

    cout << "Enter action n/e: ";
    cin >> ac;
    
    if(ac == 'n'){
      m->takeAction(ACTION_NORTH);
    }
    else if(ac == 'e'){
      m->takeAction(ACTION_EAST);
    }
  }
  while(ac == 'n' || ac == 's' || ac == 'e' || ac == 'w');
}


int main(int argc, char *argv[]){
  
  // Default parameter values.
  int s = 5;
  double p = 0.1;
  double chi = 0.5;
  int w = 1;
  double sigma = 0;
  string method = "sarsa_0";
  int randomSeed = time(0);
  string outFileName = "";

  if(!(setRunParameters(argc, argv, s, p, chi, w, sigma, method, randomSeed, outFileName))){
    options();
    return 1;
  }

  cout << "s: " << s << "\n";
  cout << "p: " << p << "\n";
  cout << "chi: " << chi << "\n";
  cout << "w: " << w << "\n";
  cout << "sigma: " << sigma << "\n";
  cout << "method: " << method << "\n";
  cout << "randomSeed: " << randomSeed << "\n";
  cout << "outFileName: " << outFileName << "\n";

  // Start train and test MDP's with same parameters and random seed.
  MDP *trainMDP = new MDP(s, p, chi, w, sigma, randomSeed);
  MDP *testMDP = new MDP(s, p, chi, w, sigma, randomSeed);
  double minimalValue = testMDP->getMinimalValue();
  double maximalValue = testMDP->getMaximalValue();
  double randomValue = testMDP->getValueRandomPolicy();

  //cout << "train Maximal Value: " << trainMDP->getMaximalValue() << "\n";
  //cout << "train Minimal Value: " << trainMDP->getMinimalValue() << "\n";
  //cout << "train Random Value: " << trainMDP->getValueRandomPolicy() << "\n";
  //cout << "test Maximal Value: " << testMDP->getMaximalValue() << "\n";
  //cout << "test Minimal Value: " << testMDP->getMinimalValue() << "\n";
  //cout << "test Random Value: " << testMDP->getValueRandomPolicy() << "\n";
  //test(testMDP);
  //test(trainMDP);
  //return 1;

  
  // Initialise agent policy.
  Agent *agent;
  if(method == "random"){
    agent = new RandomAgent(testMDP->getNumFeatures(), 2, randomSeed);
  }
  else if(method.find("sarsa") == 0){

    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	}
	else{
	  options(); return 1;
	}
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }
    agent = new SarsaLambdaAgent(testMDP->getNumFeatures(), 2, lambda, alphaInit, epsInit, randomSeed);

  }
  else if(method.find("consarsa") == 0){

    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	}
	else{
	  options(); return 1;
	}
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }
    agent = new ConSarsaLambdaAgent(testMDP->getNumFeatures(), 2, lambda, alphaInit, epsInit, randomSeed);

  }
  else if(method.find("hlsarsa") == 0){

    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	}
	else{
	  options(); return 1;
	}
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }
    agent = new HLSarsaLambdaAgent(testMDP->getNumFeatures(), 2, lambda, alphaInit, epsInit, randomSeed);

  }
  else if(method.find("expsarsa") == 0){

    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	}
	else{
	  options(); return 1;
	}
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }
    agent = new ExpSarsaLambdaAgent(testMDP->getNumFeatures(), 2, lambda, alphaInit, epsInit, randomSeed);

  }
  else if(method.find("conexpsarsa") == 0){

    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	}
	else{
	  options(); return 1;
	}
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }
    agent = new ConExpSarsaLambdaAgent(testMDP->getNumFeatures(), 2, lambda, alphaInit, epsInit, randomSeed);

  }
  else if(method.find("greedygq") == 0){


    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      pos = method.find("_", pos + 1);
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	}
	else{
	  options(); return 1;
	}
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }

    agent = new GreedyGQLambdaAgent(testMDP->getNumFeatures(), 2, lambda, alphaInit, epsInit, randomSeed);
  }
  else if(method.find("qlearning") == 0){

    double alphaInit = 1.0;
    double epsInit = 1.0;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }

    agent = new QLearningAgent(testMDP->getNumFeatures(), 2, alphaInit, epsInit, randomSeed);
  }
  else if(method.find("ce") == 0){

    int popSize = 100;
    int evalEpisodes = 2000;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      popSize = atoi((method.substr(pos + 1, method.length() - pos)).c_str());

      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	evalEpisodes = atoi((method.substr(pos + 1, method.length() - pos)).c_str());
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }

    agent = new CrossEntropyAgent(testMDP->getNumFeatures(), 2, popSize, evalEpisodes, randomSeed);
  }
  else if(method.find("cmaes") == 0){

    int popSize = 100;
    int evalEpisodes = 2000;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      popSize = atoi((method.substr(pos + 1, method.length() - pos)).c_str());

      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	evalEpisodes = atoi((method.substr(pos + 1, method.length() - pos)).c_str());
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }

    agent = new CMAESAgent(testMDP->getNumFeatures(), 2, popSize, evalEpisodes, randomSeed);
  }
  else if(method.find("ga") == 0){

    int popSize = 100;
    int evalEpisodes = 2000;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      popSize = atoi((method.substr(pos + 1, method.length() - pos)).c_str());

      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	evalEpisodes = atoi((method.substr(pos + 1, method.length() - pos)).c_str());
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }

    agent = new GeneticAlgorithmAgent(testMDP->getNumFeatures(), 2, popSize, evalEpisodes, randomSeed);
  }
  else if(method.find("hc") == 0){

    int popSize = 100;
    int evalEpisodes = 2000;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      popSize = atoi((method.substr(pos + 1, method.length() - pos)).c_str());
      
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	evalEpisodes = atoi((method.substr(pos + 1, method.length() - pos)).c_str());
      }
      else{
	options(); return 1;
      }
      
      agent = new HillClimbingAgent(testMDP->getNumFeatures(), 2, popSize, evalEpisodes, randomSeed);
    }
  }
  else if(method.find("rwg") == 0){

    int evalEpisodes = 2000;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      evalEpisodes = atoi((method.substr(pos + 1, method.length() - pos)).c_str());
    }
    else{
      options();
      return 1;
    }

    agent = new RWGAgent(testMDP->getNumFeatures(), 2, evalEpisodes, randomSeed);
  }
  else if(method.find("transfer") == 0){

    double lambda = 0;
    int evalEpisodes = 500;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());

      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	evalEpisodes = atoi((method.substr(pos + 1, method.length() - pos)).c_str());
      }
      else{
	options();
	return 1;
      }
    }
    else{
      options();
      return 1;
    }
    ////////    agent = new TransferAgent(testMDP->getNumFeatures(), 2, lambda, evalEpisodes, randomSeed);

  }
  else{
    options();
    return 1;
  }


  // How many total episodes, and how many episodes per evaluation?
  unsigned long numTotalEpisodes = 50000;
  int numEvalEpisodes = 1000;
  
  // Points (number of training episodes) after which MDP should be evaluated.
  set<int> points;
  points.insert(0);
  points.insert(2000);
  points.insert(4000);
  points.insert(6000);
  points.insert(8000);
  points.insert(10000);
  points.insert(12000);
  points.insert(14000);
  points.insert(16000);
  points.insert(18000);
  points.insert(20000);
  points.insert(22000);
  points.insert(24000);
  points.insert(26000);
  points.insert(28000);
  points.insert(30000);
  points.insert(32000);
  points.insert(34000);
  points.insert(36000);
  points.insert(38000);
  points.insert(40000);
  points.insert(42000);
  points.insert(44000);
  points.insert(46000);
  points.insert(48000);
  points.insert(50000);

  // Write output.
  stringstream output(stringstream::in | stringstream::out);
  output << "# start output\n";

  output << "# start values\n";

  for(unsigned int e = 0; e <= numTotalEpisodes; e++){
		  
    if(points.find(e) != points.end()){

      double val = evaluate(testMDP, agent, numEvalEpisodes);
      //val = (val - minimalValue) / (maximalValue - minimalValue);
      val = (val - randomValue) / (maximalValue - randomValue);
      output << e << "\t" << val << "\n";
      cout << e << "\t" << val << "\n";
    }

    vector<double> state;
    int action;
    bool term;
    double reward;

    term = false;
    state = trainMDP->getFeatures();
    while(!term){
      
      action = agent->takeAction(state);
      term = trainMDP->takeAction(action);
      reward = trainMDP->getLastReward();
      state = trainMDP->getFeatures();
      agent->update(reward, state, term);
    }
    
  }

  output << "# end values\n";

  output << "# end output\n";

  // If output file is specified, write to it; otherwise write to console.
  if(outFileName.length() > 0){
    fstream file;
    file.open(outFileName.c_str(), ios::out);
    file << output.str();
    file.close();
  }
  else{
    cout << output.str();
  }


  // Deallocate memory.
  delete agent;
  delete trainMDP;
  delete testMDP;

  return 0;
}


