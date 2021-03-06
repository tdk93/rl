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
#include "expsarsalambdaagent.h"
#include "greedygqlambdaagent.h"
#include "qlearningagent.h"
#include "doubleqlearningagent.h"

#include "crossentropyagent.h"
#include "optcrossentropyagent.h"
#include "cmaesagent.h"
#include "optcmaesagent.h"
#include "geneticalgorithmagent.h"
#include "rwgagent.h"
#include "transferagent.h"
#include "opttransferagent.h"

using namespace std;


void options(){

  cout << "Usage:\n";
  cout << "mdp-evaluate\n"; 
  cout << "\t[--s s]\n";
  cout << "\t[--p p]\n";
  cout << "\t[--chi chi]\n";
  cout << "\t[--w w]\n";
  cout << "\t[--sigma sigma]\n"; 
  cout << "\t[--method random | sarsa_lambda_alphainit_epsinit_initWeight | expsarsa_lambda_alphainit_epsinit_initWeight | greedygq_lambda_alphainit_epsinit_initWeight | qlearning_lambda_alphainit_epsinit_initWeight | ce_generations_evalepisodes | cmaes_generations_evalepisodes | optcmaes_generations_evalepisodes | ga_generations_evalepisodes | rwg_evalepisodes | transfer_lambda_alphaInit_epsInit_initWeight_transferpointepisodes_generations_evalEpisodes | opttransfer_lambda_alphaInit_epsInit_initWeight_transferpointepisodes_generations_evalEpisodes]\n";
  cout << "\t[--totalEpisodes totalEpisodes]\n";
  cout << "\t[--displayInterval displayInterval]\n";
  cout << "\t[--randomSeed randomSeed]\n";
  cout << "\t[--outFile outFile]\n";
}


//  Read command line arguments, and set the ones that are passed (the others remain default.)
bool setRunParameters(int argc, char *argv[], int &s, double &p, double &chi, int &w, double &sigma, string &method, unsigned long int &totalEpisodes, unsigned long int &displayInterval, int &randomSeed, string &outFileName){

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
    else if(string(argv[ctr]) == "--totalEpisodes"){
      if(ctr == (argc - 1)){
	return false;
      }
      totalEpisodes = atol(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--displayInterval"){
      if(ctr == (argc - 1)){
	return false;
      }
      displayInterval = atol(string(argv[ctr + 1]).c_str());
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
  string method = "sarsa_0_1.0_1.0";
  unsigned long int totalEpisodes = 50000;
  unsigned long int displayInterval = 2000;
  int randomSeed = time(0);
  string outFileName = "";


  if(!(setRunParameters(argc, argv, s, p, chi, w, sigma, method, totalEpisodes, displayInterval, randomSeed, outFileName))){
    options();
    return 1;
  }

  cout << "s: " << s << "\n";
  cout << "p: " << p << "\n";
  cout << "chi: " << chi << "\n";
  cout << "w: " << w << "\n";
  cout << "sigma: " << sigma << "\n";
  cout << "method: " << method << "\n";
  cout << "total episodes: " << totalEpisodes << "\n";
  cout << "display interval: " << displayInterval << "\n";
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
  //  test(testMDP);
  //test(trainMDP);
  double averageSwitches = testMDP->getSwitchFrequency();
  cout << "Average Switches: " << averageSwitches << "\n";

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
    double initWeight = 0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	  pos = method.find("_", pos + 1);
	  if(pos != string::npos){
	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
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
    }
    else{
      options(); return 1;
    }
    agent = new SarsaLambdaAgent(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, randomSeed);

  }
  else if(method.find("expsarsa") == 0){

    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;
    double initWeight = 0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	  pos = method.find("_", pos + 1);
	  if(pos != string::npos){
	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
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
    }
    else{
      options(); return 1;
    }
    agent = new ExpSarsaLambdaAgent(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, randomSeed);

  }
  else if(method.find("greedygq") == 0){


    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;
    double initWeight = 0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      pos = method.find("_", pos + 1);
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	  pos = method.find("_", pos + 1);
	  if(pos != string::npos){
	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
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
    }
    else{
      options(); return 1;
    }

    agent = new GreedyGQLambdaAgent(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, randomSeed);
  }
  else if(method.find("qlearning") == 0){

    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;
    double initWeight = 0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	  pos = method.find("_", pos + 1);
	  if(pos != string::npos){
	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
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
    }
    else{
      options(); return 1;
    }

    agent = new QLearningAgent(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, randomSeed);
  }
  else if(method.find("doubleqlearning") == 0){

    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;
    double initWeight = 0;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	  pos = method.find("_", pos + 1);
	  if(pos != string::npos){
	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
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
    }
    else{
      options(); return 1;
    }

    agent = new DoubleQLearningAgent(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, randomSeed);
  }
  else if(method.find("ce") == 0){

    int generations = 100;
    int evalEpisodes = 2000;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      generations = atoi((method.substr(pos + 1, method.length() - pos)).c_str());

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

    agent = new CrossEntropyAgent(testMDP->getNumFeatures(), 2, totalEpisodes, generations, evalEpisodes, randomSeed);
  }
  else if(method.find("optce") == 0){

    int generations = 100;
    int evalEpisodes = 2000;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      generations = atoi((method.substr(pos + 1, method.length() - pos)).c_str());

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

    agent = new OptCrossEntropyAgent(testMDP->getNumFeatures(), 2, totalEpisodes, generations, evalEpisodes, randomSeed);
  }
  else if(method.find("cmaes") == 0){

    int generations = 100;
    int evalEpisodes = 2000;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      generations = atoi((method.substr(pos + 1, method.length() - pos)).c_str());

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

    agent = new CMAESAgent(testMDP->getNumFeatures(), 2, totalEpisodes, generations, evalEpisodes, randomSeed);
  }
  else if(method.find("optcmaes") == 0){

    int generations = 100;
    int evalEpisodes = 2000;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      generations = atoi((method.substr(pos + 1, method.length() - pos)).c_str());

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

    agent = new OptCMAESAgent(testMDP->getNumFeatures(), 2, totalEpisodes, generations, evalEpisodes, randomSeed);
  }
  else if(method.find("ga") == 0){

    int generations = 100;
    int evalEpisodes = 2000;
    unsigned int pos = method.find("_");
    if(pos != string::npos){
      generations = atoi((method.substr(pos + 1, method.length() - pos)).c_str());

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

    agent = new GeneticAlgorithmAgent(testMDP->getNumFeatures(), 2, totalEpisodes, generations, evalEpisodes, randomSeed);
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
    double alphaInit = 1.0;
    double epsInit = 1.0;
    double initWeight = 0;

    unsigned long int transferPointEpisodes = 25000;

    int generations = 100;
    int evalEpisodes = 2000;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	  pos = method.find("_", pos + 1);
	  if(pos != string::npos){
	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	    pos = method.find("_", pos + 1);
	    if(pos != string::npos){
	      transferPointEpisodes = atol((method.substr(pos + 1, method.length() - pos)).c_str());
	      pos = method.find("_", pos + 1);
	      if(pos != string::npos){
		generations = atoi((method.substr(pos + 1, method.length() - pos)).c_str());
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
      }
      else{
	options(); return 1;
      }
    }
    else{
      options(); return 1;
    }
    
    agent = new TransferAgent(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, transferPointEpisodes, generations, evalEpisodes, randomSeed);
  }
  else if(method.find("opttransfer") == 0){

    double lambda = 0;
    double alphaInit = 1.0;
    double epsInit = 1.0;
    double initWeight = 0;

    unsigned long int transferPointEpisodes = 25000;

    int generations = 100;
    int evalEpisodes = 2000;

    unsigned int pos = method.find("_");
    if(pos != string::npos){
      lambda = atof((method.substr(pos + 1, method.length() - pos)).c_str());
      pos = method.find("_", pos + 1);
      if(pos != string::npos){
	alphaInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	pos = method.find("_", pos + 1);
	if(pos != string::npos){
	  epsInit = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	  pos = method.find("_", pos + 1);
	  if(pos != string::npos){
	    initWeight = atof((method.substr(pos + 1, method.length() - pos)).c_str());
	    pos = method.find("_", pos + 1);
	    if(pos != string::npos){
	      transferPointEpisodes = atol((method.substr(pos + 1, method.length() - pos)).c_str());
	      pos = method.find("_", pos + 1);
	      if(pos != string::npos){
		generations = atoi((method.substr(pos + 1, method.length() - pos)).c_str());
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
      }
      else{
	options(); return 1;
      }
    }
  else{
      options(); return 1;
    }
    
    agent = new OptTransferAgent(testMDP->getNumFeatures(), 2, totalEpisodes, initWeight, lambda, alphaInit, epsInit, transferPointEpisodes, generations, evalEpisodes, randomSeed);
  }
  else{
    options();
    return 1;
  }


  // How many episodes per evaluation?
  int numEvalEpisodes = 1000;
  
  // Points (number of training episodes) after which MDP should be evaluated.
  set<int> points;

  unsigned long int nextPoint = 0;
  while(nextPoint <= totalEpisodes){
    points.insert(nextPoint);
    nextPoint += displayInterval;
  }

  // Write output.
  stringstream output(stringstream::in | stringstream::out);
  output << "# start output\n";

  output << "# start values\n";

  for(unsigned int e = 0; e <= totalEpisodes; e++){
		  
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


