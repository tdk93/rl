#include "MDP.h"

MDP::MDP(int side, double featureFraction, double stateNoiseSigma, double actionNoiseProb){

  this->side = side;
  this->stateNoiseSigma = stateNoiseSigma;
  this->actionNoiseProb = actionNoiseProb;
  
  numStates = (side * side);
  numTermini = (int)((TERMINI_FRACTION * (double)numStates) + 0.5);
  numFeatures = (int)((featureFraction * (numStates - numTermini)) + 0.5);

  if(numFeatures > (numStates - numTermini)){
    numFeatures = numStates - numTermini;
  }
 

  cell.clear();
  for(int i = 0; i < side; i++){
    for(int j = 0; j < side; j++){

      Cell c;
      c.x = (double)(i + 0.5);
      c.y = -(double)(j + 0.5);

      cell.push_back(c);
    }
  }


  srand48(time(0));

  fixTerminalStates();
  fixRewards();
  computeOptimalValueAndPolicy();
  fixFeatures();
  reset();

  lastReward = 0;
}

MDP::~MDP(){
}

void MDP::fixTerminalStates(){

  for(int i = 0; i < numStates; i++){
    cell[i].isTerminal = false;
  }
  
  for(int i = 0; i < numTermini; i++){
    
    int s = 0;
    do{
      s = (int)(drand48() * numStates) % numStates;
    }while(cell[s].isTerminal);
    
    cell[s].isTerminal = true;
  }
  
}

void MDP::fixRewards(){

  for(int i = 0; i < numStates; i++){
    cell[i].reward = drand48() * -1.0;
  }

}

void MDP::fixFeatures(){

  for(int i = 0; i < numStates; i++){
    cell[i].hasFeature = false;
  }
  
  for(int i = 0; i < numFeatures; i++){
    
    int s = 0;
    do{
      s = (int)(drand48() * numStates) % numStates;
    }while(cell[s].hasFeature || cell[s].isTerminal);
    
    cell[s].hasFeature = true;
  }
 
  for(int i = 0; i < numStates; i++){

    if(cell[i].isTerminal){
      continue;
    }

    cell[i].featureValue.clear();

    for(int j = 0; j < numStates; j++){

      if(cell[j].hasFeature){

	double distanceX = cell[j].x - cell[i].x;
	double distanceY = cell[j].y - cell[i].y;
	double distanceSquared= (distanceX * distanceX) + (distanceY * distanceY);

	double value = exp(-distanceSquared / 2.0 * FEATURE_SD * FEATURE_SD);

	cell[i].featureValue.push_back(value);
      }
    }
  }
}

void MDP::reset(){

  do{
    currentState = (int)(drand48() * numStates) % numStates;
  }while(cell[currentState].isTerminal);

}

int MDP::getNumFeatures(){
  
  return numFeatures;
}

std::vector<double> MDP::getFeatures(){

  std::vector<double> f = std::vector<double>(cell[currentState].featureValue);
  return f;
}

int MDP::getNeighbour(int s, int d){
  
  int n = 0;

  if(d == DIR_NORTH){
    n = s - side;
    if(n < 0){
      n = s;
    }
    return n;
  }
  else if(d == DIR_SOUTH){
    n = s + side;
    if(n >= numStates){
      n = s;
    }
    return n;
  }
  else if(d == DIR_EAST){
    n = s + 1;
    if(n % side == 0){
      n = s;
    }
    return n;
  }
  else if(d == DIR_WEST){
    n = s - 1;
    if(s % side == 0){
      n = s;
    }
    return n;
  }

  //Error.
  return -1;
}


bool MDP::takeAction(int action){

  int n = 0;

  int takenAction = action;

  double r = drand48();

  if(r < actionNoiseProb){
    
    while(takenAction == action){
      takenAction = (int)(drand48() * 4) % 4;
    }
  }

  n = getNeighbour(currentState, takenAction);

  currentState = n;
  bool term = cell[currentState].isTerminal;

  lastReward = cell[currentState].reward;
  
  if(term){

    reset();
  }

  return term;
}

void MDP::display(int displayType){

  int s = 0;
  for(int i = 0; i < side; i++){
    for(int j = 0; j < side; j++){

      std::stringstream dis;
      int len;

      if(displayType == DISPLAY_FEATURES){
	dis << (int)(cell[s].hasFeature);
	len = 1;
      }
      else if(displayType == DISPLAY_TERMINAL){
	dis << (int)(cell[s].isTerminal);
	len = 1;
      }
      else if(displayType == DISPLAY_REWARDS){
	dis << cell[s].reward;
	len = 5;
      }
      else if(displayType == DISPLAY_ACTIONS){

	if(cell[s].optimalAction == ACTION_NORTH){
	  dis << "^";
	}
	else if(cell[s].optimalAction == ACTION_SOUTH){
	  dis << "v";
	}
	else if(cell[s].optimalAction == ACTION_EAST){
	  dis << ">";
	}
	else if(cell[s].optimalAction == ACTION_WEST){
	  dis << "<";
	}

	len = 1;
      }
      else{//displayType == DISPLAY_VALUES
	dis << cell[s].optimalValue;
	len = 5;
      }

      string str = dis.str();
      for(int l = 0; l < len; l++){
	if(l < str.length()){
	  cout << str.at(l);
	}
	else{
	  cout << " ";
	}

      }

      if(s == currentState){
	cout << "*";
      }
      else{
	cout << "_";
      }

      s++;
    }
    cout << "\n";
  }

}

double MDP::getLastReward(){

  return lastReward;
}

std::vector<double> MDP::computeOptimalDeterministicValue(){

  std::vector<double> value;

  for(int i = 0; i < numStates; i++){

    if(cell[i].isTerminal){
      value.push_back(0);
    }
    else{
      value.push_back(-100000.0);//minus infinity
    }
  }

  bool converged;
  do{

    converged = true;

    for(int i = 0; i < numStates; i++){

      if(cell[i].isTerminal){
	continue;
      }

      int north = getNeighbour(i, DIR_NORTH);
      int south = getNeighbour(i, DIR_SOUTH);
      int east = getNeighbour(i, DIR_EAST);
      int west = getNeighbour(i, DIR_WEST);

      double valNorth = cell[north].reward + value[north];
      double valSouth = cell[south].reward + value[south];
      double valEast = cell[east].reward + value[east];
      double valWest = cell[west].reward + value[west];

      if(valNorth > value[i]){
	value[i] = valNorth;
	converged = false;
      }
      if(valSouth > value[i]){
	value[i] = valSouth;
	converged = false;
      }
      if(valEast > value[i]){
	value[i] = valEast;
	converged = false;
      }
      if(valWest > value[i]){
	value[i] = valWest;
	converged = false;
      }

    }

  }
  while(!converged);

  return value;
}


std::vector<double> MDP::computeValue(std::vector<int> policy){

  std::vector<double> value;

  for(int i = 0; i < numStates; i++){
    value.push_back(0);
  }

  double tolerance = 1.0e-6;
  double error;

  do{
    
    error = 0;
    
    for(int i = 0; i < numStates; i++){
      
      if(cell[i].isTerminal){
	continue;
      }
      
      double newValue = 0;
      
      int nextState = getNeighbour(i, policy[i]);
      newValue = (1.0 - actionNoiseProb) * (cell[nextState].reward + value[nextState]);
      
      int north = getNeighbour(i, DIR_NORTH);
      int south = getNeighbour(i, DIR_SOUTH);
      int east = getNeighbour(i, DIR_EAST);
      int west = getNeighbour(i, DIR_WEST);
      
      newValue += (actionNoiseProb * 0.25) * (cell[north].reward + value[north]);
      newValue += (actionNoiseProb * 0.25) * (cell[south].reward + value[south]);
      newValue += (actionNoiseProb * 0.25) * (cell[east].reward + value[east]);
      newValue += (actionNoiseProb * 0.25) * (cell[west].reward + value[west]);
      
      error += fabs(newValue - value[i]);
      value[i] = newValue;
    }
    
    //    cout << "Error: " << error << "\n";
  }
  while(error > tolerance);

  return value;
}

std::vector<int> MDP::obtainPolicy(std::vector<double> value){

  std::vector<int> policy;

  for(int i = 0; i < numStates; i++){
    policy.push_back(0);
  }

  for(int i = 0; i < numStates; i++){
    
    if(cell[i].isTerminal){
      continue;
    }
    
    int north = getNeighbour(i, DIR_NORTH);
    int south = getNeighbour(i, DIR_SOUTH);
    int east = getNeighbour(i, DIR_EAST);
    int west = getNeighbour(i, DIR_WEST);
    
    double northRelValue = (actionNoiseProb * 0.25) * (cell[north].reward + value[north]);
    double southRelValue = (actionNoiseProb * 0.25) * (cell[south].reward + value[south]);
    double eastRelValue = (actionNoiseProb * 0.25) * (cell[east].reward + value[east]);
    double westRelValue = (actionNoiseProb * 0.25) * (cell[west].reward + value[west]);
    
    double bestValue = northRelValue;
    policy[i] = ACTION_NORTH;
    if(southRelValue > bestValue){
      bestValue = southRelValue;
      policy[i] = ACTION_SOUTH;
    }
    if(eastRelValue > bestValue){
      bestValue = eastRelValue;
      policy[i] = ACTION_EAST;
    }
    if(westRelValue > bestValue){
      bestValue = westRelValue;
      policy[i] = ACTION_WEST;
    }
  }

  return policy;
}


void MDP::computeOptimalValueAndPolicy(){

  std::vector<int> policy;

  std::vector<double> value;

  value = computeOptimalDeterministicValue();
  policy = obtainPolicy(value);

  if(actionNoiseProb >= 1.0e-6){

    bool valueChanged;
    double EPS = 1.0e-4;
    
    int ctr = 0;
    
    int noiseLevels = 100;
    double noise;
    
    do{
      
      valueChanged = false;
      
      std::vector<double> tempValue = computeValue(policy);
      for(int i = 0; i < numStates; i++){
	
	if(fabs(value[i] - tempValue[i]) > EPS){
	  valueChanged = true;
	}
	
	value[i] = tempValue[i];
      }
      
      policy = obtainPolicy(value);
      
      //      cout << "Value Changed " << ctr++ << "\n";
    }
    while(valueChanged);
    
  }
  
  
  for(int i = 0; i < numStates; i++){
    
    cell[i].optimalValue = value[i];
    cell[i].optimalAction = policy[i];
  }
}

double MDP::getOptimalValue(){

  double v = 0;

  for(int i = 0; i < numStates; i++){
    
    v += cell[i].optimalValue;
  }

  v /= (numStates - numTermini);
 
  return v;
}
