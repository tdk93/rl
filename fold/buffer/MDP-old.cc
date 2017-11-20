#include "MDP.h"

MDP::MDP(int side, double featureFraction, double stateNoiseSigma, double actionNoiseProb){

  this->side = side;
  
  numTermini = (int)((TERMINI_FRACTION * (double)side * side) + 0.5);
  //  numTermini = 1;
  //  cout << "NT: " << numTermini << "\n";

  numStates = (side * side);
  //  cout << "NS: " << numStates << "\n";

  numFeatures = (int)((featureFraction * (numStates - numTermini)) + 0.5);
  if(numFeatures > (numStates - numTermini)){
    numFeatures = numStates - numTermini;
  }
  //  cout << "NF: " << numFeatures << "\n";
 
  srand48(time(0));

  fixTerminalStates();

  fixFeatures();

  reset();

  lastReward = 0;

  this->stateNoiseSigma = stateNoiseSigma;
  this->actionNoiseProb = actionNoiseProb;

}

MDP::~MDP(){
}

void MDP::fixTerminalStates(){

  bool flag = true;

  do{
    
    for(int i = 0; i < numStates; i++){
      terminal[i] = false;
    }
    
    //    srand48(time(0));
    
    flag = false;
    
    for(int i = 0; i < numTermini; i++){
      
      int s = 0;
      do{
	s = (int)(drand48() * numStates) % numStates;
      }while(terminal[s]);
      
      terminal[s] = true;
    }
    
    for(int i = 0; i < numStates; i++){
      
      if(terminal[getNeighbour(i, DIR_NORTH)]){
	if(terminal[getNeighbour(i, DIR_SOUTH)]){
	  if(terminal[getNeighbour(i, DIR_EAST)]){
	    if(terminal[getNeighbour(i, DIR_WEST)]){
	      flag = true;
	    }
	  }
	}
      }
    }
  }
  while(flag);

}

void MDP::fixFeatures(){

  for(int i = 0; i < numStates; i++){
    feature[i] = -1;
  }

  //  srand48(time(0));

  for(int i = 0; i < numFeatures; i++){

    int s = 0;

    do{
      s = (int)(drand48() * numStates) % numStates;
    }while(feature[s] != -1 || terminal[s]);

    feature[s] = i;
  }

  bool flag;

  do{
    
    flag = false;

    for(int i = 0; i < numStates; i++){
      
      if(!terminal[i] && feature[i] == -1){
	flag = true;

	int d= (int)(drand48() * 4) % 4;
	int s = getNeighbour(i, d);

	if(!terminal[s] && feature[s] != -1){
	  feature[i] = feature[s];
	}

      }

    }

  }
  while(flag);

}

void MDP::reset(){

  //  srand48(time(0));

  do{
    currentState = (int)(drand48() * numStates) % numStates;
  }while(terminal[currentState]);

}

int MDP::getNumFeatures(){
  
  return numFeatures;
}

void MDP::getFeatures(double f[]){

  double p[numStates];

  int currentRow = currentState / side;
  int currentColumn = currentState % side;

  double sum = 0;
  for(int i = 0; i < numStates; i++){

    if(terminal[i]){
      p[i] = 0;
    }
    else{
      
      int row = i / side;
      int column = i % side;
      
      double distanceSquare = (row - currentRow) * (row - currentRow);
      distanceSquare += (column  - currentColumn) * (column - currentColumn);
      
      if(stateNoiseSigma == 0){
	if(distanceSquare == 0){
	  p[i] = 1.0;
	}
	else{
	  p[i] = 0;
	}
      }
      else{
	p[i] = exp(-distanceSquare / (2.0 * stateNoiseSigma * stateNoiseSigma));
      }
    }
    
    sum += p[i];
  }

  for(int i = 0; i < numStates; i++){
    p[i] /= sum;
  }

  //  srand48(time(0));

  double prob = drand48();
  int ctr = 0;
  while(prob > 0 & ctr < numStates){
    prob -= p[ctr++];
  }

  int chosenState = ctr - 1;

  for(int i = 0; i < numFeatures; i++){
    if(i == feature[chosenState]){
      f[i] = 1.0;
    }
    else{
      f[i] = 0;
    }
  }
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

  //  srand48(time(0));
  
  double r = drand48();

  if(r < actionNoiseProb){
    
    while(takenAction == action){
      takenAction = (int)(drand48() * 4) % 4;
    }
  }

  if(takenAction == ACTION_NORTH){
    n = currentState - side;
    if(n < 0){
      n = currentState;
    }
  }
  else if(takenAction == ACTION_SOUTH){
    n = currentState + side;
    if(n >= numStates){
      n = currentState;
    }
  }
  else if(takenAction == ACTION_EAST){
    n = currentState + 1;
    if(n % side == 0){
      n = currentState;
    }
  }
  else if(takenAction == ACTION_WEST){
    n = currentState - 1;
    if(currentState % side == 0){
      n = currentState;
    }
  }

  currentState = n;
  bool term = terminal[currentState];
  
  if(term){
    reset();
  }

  lastReward = -1.0;

  return term;
}

void MDP::display(){

  int s = 0;
  for(int i = 0; i < side; i++){
    for(int j = 0; j < side; j++){

      //int p = s;
      int p = feature[s];
      if(terminal[s]){
	cout << "TT";
      }
      else{
	if(p < 10){
	  cout << "0";
	}
	cout << p;
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

  double f[numFeatures];
  getFeatures(f);

  cout << "Active Feature: ";
  for(int i = 0; i < numFeatures; i++){
    if(f[i]){
      cout << i << "\n";
    }
  }
}

double MDP::getLastReward(){

  return lastReward;
}

double MDP::computeDeterministicOptimalValue(){

  double sum = 0;

  for(int i = 0; i < numStates; i++){

    if(!terminal[i]){

      double minDis = 10000.0;//infinity
      int row = i / side;
      int column = i % side;

      for(int j = 0; j < numStates; j++){
	
	if(terminal[j]){

	  int termRow = j / side;
	  int termColumn = j % side;

	  double dis = fabs(row - termRow) + fabs(column - termColumn);

	  if(dis < minDis){
	    minDis = dis;
	  }
	}
      }

      sum += minDis;
    }
  }

  double val = -(sum / (numStates - numTermini));

  return val;
}


