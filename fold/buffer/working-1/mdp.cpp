#include "mdp.h"

MDP::MDP(const int &s, const double &p, const double &chi, const int &minFeatureWidth, const double &sigma, const int &randomSeed){
  
  // Copy over arguments.
  this->side = s;
  this->actionNoiseProb = p;

  this->minFeatureWidth = minFeatureWidth;
  
  // States on the east and north borders of the square grid are terminal.
  numStates = (side * side);
  numTermini = (2 * side) - 1;
  
  // Compute number of features based on featureFraction.
  // chi, w, and side are related by an inequality -- which must
  // hold if all the cells are to be covered.
  numFeatures = (int)((chi * (numStates - numTermini)) + 0.5);
  if(numFeatures > (numStates - numTermini)){
    numFeatures = numStates - numTermini;
  }
  
  // Fill x and y values for cells.
  cell.clear();
  for(int i = 0; i < side; i++){
    for(int j = 0; j < side; j++){
      
      Cell c;
      c.x = (double)(i + 0.5);
      c.y = -(double)(j + 0.5);
      
      cell.push_back(c);
    }
  }
  
  // Initialise random number generator.
  ran = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(ran, randomSeed);
  
  // Fix terminal states and rewards.
  fixTerminalStates();
  fixRewards();
  
  // Compute maximal and minimal value functions and policies.
  computeOptimalValueAndPolicy(true);
  computeOptimalValueAndPolicy(false);

  // Allot features to cells.
  fixFeatures();

  reset();
  
  lastReward = 0;


  if(sigma > EPS){
    stateNoiseWidthX = gsl_ran_gaussian(ran, sigma);
    stateNoiseWidthY = gsl_ran_gaussian(ran, sigma);
  }
  else{
    stateNoiseWidthX = 0;
    stateNoiseWidthY = 0;
  }

  //  cout << "sx: " << stateNoiseWidthX << "\n";
  //  cout << "sy: " << stateNoiseWidthY << "\n";
}


MDP::~MDP(){
  
  gsl_rng_free(ran);
}


void MDP::fixTerminalStates(){
  
  for(int i = 0; i < numStates; i++){
    
    int row = i / side;
    int col = i % side;
    
    if(row == 0 || col == (side - 1)){
      cell[i].isTerminal = true;
    }
    else{
      cell[i].isTerminal = false;
    }
  }

}


void MDP::fixRewards(){
  
  for(int i = 0; i < numStates; i++){
    cell[i].reward = gsl_rng_uniform(ran);
  }
}

int MDP::getNumCellsCovered(){

  int numCellsCovered = 0;
  
  for(int i = 0; i < numStates; i++){

    if(cell[i].isTerminal){
      continue;
    }

    bool cellCovered = false;
    
    for(int j = 0; j < numStates; j++){
      
      bool b1 = cell[j].hasFeature;
      bool b2 = fabs(cell[i].x - cell[j].x) < (cell[j].featureWidth * 0.5);
      bool b3 = fabs(cell[i].y - cell[j].y) < (cell[j].featureWidth * 0.5);
      
      if(b1 && b2 && b3){
	cellCovered = true;
	break;
      }
    }
    
    if(cellCovered){
      numCellsCovered++;
    }
    
  }
  
  return numCellsCovered;
}


void MDP::fixFeatures(){

  for(int i = 0; i < numStates; i++){
    cell[i].hasFeature = false;
  }

  int featuresCovered = 0;

  // Place centres regularly to just sover the entire region.
  for(int i = 0; i < numStates; i++){
    
    int row = i / side;
    int col = i % side;
    
    if(minFeatureWidth == 1 && !(cell[i].isTerminal)){
      cell[i].hasFeature = true;
      cell[i].featureWidth = minFeatureWidth;
      
      featuresCovered++;
    }
    else if((row % minFeatureWidth) == (minFeatureWidth / 2) && (col % minFeatureWidth) == (minFeatureWidth / 2) && !(cell[i].isTerminal)){
      cell[i].hasFeature = true;
      cell[i].featureWidth = minFeatureWidth;
      
      featuresCovered++;
    }
  }

  //  display(DISPLAY_FEATURES);

  // Place remaining centres randomly on non-terminal cells that
  // are not alreadu centres.
  while(featuresCovered < numFeatures){
    
    int s = ((int)(gsl_rng_uniform(ran) * numStates)) % numStates;
    if(!(cell[s].isTerminal || cell[s].hasFeature)){
      
      cell[s].hasFeature = true;
      cell[s].featureWidth = minFeatureWidth;
      featuresCovered++;
    }
  }
  

  // Swap cells (centres) multiple times to scramble them up.
  int numSwaps = 10000;
  for(int t = 0; t < numSwaps;){
    
    int s1 = ((int)(gsl_rng_uniform(ran) * numStates)) % numStates;
    int s2 = ((int)(gsl_rng_uniform(ran) * numStates)) % numStates;
    if(!(cell[s1].isTerminal) && !(cell[s2]).isTerminal){
      
      bool tempHasFeature = cell[s1].hasFeature;
      int tempFeatureWidth = cell[s1].featureWidth;
      
      cell[s1].hasFeature = cell[s2].hasFeature;
      cell[s1].featureWidth = cell[s2].featureWidth;
      
      cell[s2].hasFeature = tempHasFeature;
      cell[s2].featureWidth = tempFeatureWidth;
      
      t++;
    }
    
  }
  

  // Set the featureValue and distance vector corresponding to
  // each cell.
  for(int i = 0; i < numStates; i++){

    cell[i].featureValue.clear();

    for(int j = 0; j < numStates; j++){

      if(cell[j].hasFeature){

	double disX = fabs(cell[j].x - cell[i].x);
	double disY = fabs(cell[j].y - cell[i].y);
	double dis = sqrt((disX * disX) + (disY * disY));

	bool b1 = fabs(cell[j].x - cell[i].x) < (cell[j].featureWidth * 0.5);
	bool b2 = fabs(cell[j].y - cell[i].y) < (cell[j].featureWidth * 0.5);

	double value = 0;
	if(b1 && b2){
	  value = 1.0;
	}

	cell[i].featureValue.push_back(value);
	cell[i].distance.push_back(dis);

      }
    }
  }

}


void MDP::reset(){

  do{
    
    currentState = (int)(gsl_rng_uniform(ran) * numStates) % numStates;
  }
  while(cell[currentState].isTerminal);
  
  numStepsThisEpisode = 0;

}


int MDP::getNumFeatures(){
  
  return numFeatures;
}

vector<double> MDP::getFeatures(){
  
  int state = -1;
  int x = -1;
  int y = -1;
  
  while(x < 1 || y < 0 || x >= side || y >= side - 1){
    
    x = cell[currentState].x + (int)(gsl_rng_uniform(ran) * stateNoiseWidthX);
    y = cell[currentState].y + (int)(gsl_rng_uniform(ran) * stateNoiseWidthY);
    y = -y;
    
    //cout << "currentXY: (" << cell[currentState].x << ", " << cell[currentState].y << "), XY: " << x << ", " << y << ")\n";
  }

  state = (x * side) + y;
  return cell[state].featureValue;

}


int MDP::getNeighbour(const int &s, const int &d){
  
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


bool MDP::takeAction(const int &action){

  int takenAction = 0;
  int n = 0;

  if(action == ACTION_NORTH){

    if(gsl_rng_uniform(ran) < actionNoiseProb){
      takenAction = ACTION_EAST;
      n = getNeighbour(currentState, DIR_EAST);
    }
    else{
      takenAction = ACTION_NORTH;
      n = getNeighbour(currentState, DIR_NORTH);
    }

  }
  else if(action == ACTION_EAST){

    if(gsl_rng_uniform(ran) < actionNoiseProb){
      takenAction = ACTION_NORTH;
      n = getNeighbour(currentState, DIR_NORTH);
    }
    else{
      takenAction = ACTION_EAST;
      n = getNeighbour(currentState, DIR_EAST);
    }

  }

  currentState = n;
  bool term = cell[currentState].isTerminal;

  lastReward = cell[currentState].reward;

  numStepsThisEpisode++;

  if(term){

    reset();
  }

  return term;
}


double MDP::getLastReward(){

  return lastReward;
}


vector<double> MDP::computeValue(const vector<int> &policy){

  vector<double> value;
  value.resize(numStates, 0);

  double error;

  do{
    
    error = 0;
    
    for(int i = 0; i < numStates; i++){
      
      if(cell[i].isTerminal){
	continue;
      }
      
      double newValue = 0;
      
      int north = getNeighbour(i, DIR_NORTH);
      int east = getNeighbour(i, DIR_EAST);
      
      double n = cell[north].reward + value[north];
      double e = cell[east].reward + value[east];
      double northValue = (1.0 - actionNoiseProb) * n + actionNoiseProb * e;
      double eastValue = (1.0 - actionNoiseProb) * e + actionNoiseProb * n;

      if(policy[i] == ACTION_NORTH){
	newValue = northValue;
      }
      else if(policy[i] == ACTION_EAST){
	newValue = eastValue;
      }
    
      error += fabs(newValue - value[i]);
      value[i] = newValue;
    }
    
    //    cout << "Error: " << error << "\n";
  }
  while(error > EPS);

  return value;
}


vector<int> MDP::obtainPolicy(const vector<double> &value){

  vector<int> policy;

  for(int i = 0; i < numStates; i++){
    policy.push_back(ACTION_EAST);
  }

  for(int i = 0; i < numStates; i++){
    
    if(cell[i].isTerminal){
      continue;
    }
    
    int north = getNeighbour(i, DIR_NORTH);
    int east = getNeighbour(i, DIR_EAST);
    
    double n = cell[north].reward + value[north];
    double e = cell[east].reward + value[east];
    double northValue = (1.0 - actionNoiseProb) * n + actionNoiseProb * e;
    double eastValue = (1.0 - actionNoiseProb) * e + actionNoiseProb * n;

    double bestValue = northValue;
    policy[i] = ACTION_NORTH;
    if((eastValue > bestValue) || ((fabs(eastValue - northValue) < EPS) && (gsl_rng_uniform(ran) < 0.5))){
      bestValue = eastValue;
      policy[i] = ACTION_EAST;
    }
  }

  return policy;
}


void MDP::computeOptimalValueAndPolicy(const bool &maximal){

  vector<int> policy;
  vector<double> value;

  if(!maximal){
    for(int i = 0; i < numStates; i++){
      cell[i].reward = -cell[i].reward;
    }
  }

  value.resize(numStates, 0);
  policy = obtainPolicy(value);

  bool valueChanged;
  
  do{
    
    valueChanged = false;
    
    vector<double> tempValue = computeValue(policy);
    for(int i = 0; i < numStates; i++){
      
      if(fabs(value[i] - tempValue[i]) > EPS){
	valueChanged = true;
      }
      
      value[i] = tempValue[i];
    }
    
    policy = obtainPolicy(value);
    
  }
  while(valueChanged);
  
  
  if(!maximal){
    for(int i = 0; i < numStates; i++){
      cell[i].reward = -cell[i].reward;
    }
    
    value = computeValue(policy);
    
    for(int i = 0; i < numStates; i++){
      cell[i].minimalValue = value[i];
      cell[i].minimalAction = policy[i];
    }
    
  }
  else{
    
    for(int i = 0; i < numStates; i++){
      cell[i].maximalValue = value[i];
      cell[i].maximalAction = policy[i];
    }
  }

}


double MDP::getMaximalValue(){

  double v = 0;

  for(int i = 0; i < numStates; i++){
    
    v += cell[i].maximalValue;
  }

  v /= (numStates - numTermini);
 
  return v;
}


double MDP::getMinimalValue(){

  double v = 0;

  for(int i = 0; i < numStates; i++){
    
    v += cell[i].minimalValue;
  }

  v /= (numStates - numTermini);
 
  return v;
}

double MDP::getValueRandomPolicy(){

  vector<double> randomValue;
  randomValue.resize(numStates, 0);

  bool changed = true;
  while(changed){

    changed = false;

    for(int i = 0; i < numStates; i++){

      if(!(cell[i].isTerminal)){
	
	int north = getNeighbour(i, DIR_NORTH);
	int east = getNeighbour(i, DIR_EAST);
    
	double n = cell[north].reward + randomValue[north];
	double e = cell[east].reward + randomValue[east];

	double newRandomValue = 0.5 * (n + e);
	if(fabs(newRandomValue - randomValue[i]) > EPS){
	  changed = true;
	}
	
	randomValue[i] = newRandomValue;
      }

    }
  }

  double v = 0;

  for(int i = 0; i < numStates; i++){
    
    v += randomValue[i];
  }

  v /= (numStates - numTermini);
 
  return v;
}


void MDP::display(const int &displayType){

  unsigned int len;
  int s = 0;

  cout << "\n";
  switch(displayType){

  case DISPLAY_FEATURES: 
    cout << "Features:\n";
    len = 2;//6;
    for(int i = 0; i < side; i++){
      for(int j = 0; j < side; j++){
	stringstream dis;
	dis << (cell[s].hasFeature? "X" : ".");// << ", " << cell[s].featureWidth;
	string str = dis.str();
	for(unsigned int l = 0; l < len; l++){
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
    break;

  case DISPLAY_TERMINAL: 
    cout << "Terminal States:\n";
    len = 1;
    for(int i = 0; i < side; i++){
      for(int j = 0; j < side; j++){
	stringstream dis;
	dis << (int)(cell[s].isTerminal);
	string str = dis.str();
	for(unsigned int l = 0; l < len; l++){
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
    break;

  case DISPLAY_REWARDS: 
    cout << "Rewards:\n";
    len = 5;
    for(int i = 0; i < side; i++){
      for(int j = 0; j < side; j++){
	stringstream dis;
	dis << cell[s].reward;
	string str = dis.str();
	for(unsigned int l = 0; l < len; l++){
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
    break;

  case DISPLAY_MAXIMAL_ACTIONS: 
    cout << "Maximal Actions:\n";
    len = 1;
    for(int i = 0; i < side; i++){
      for(int j = 0; j < side; j++){
	stringstream dis;
	if(cell[s].maximalAction == ACTION_NORTH){
	  dis << "^";
	}
	else if(cell[s].maximalAction == ACTION_EAST){
	  dis << ">";
	}
	string str = dis.str();
	for(unsigned int l = 0; l < len; l++){
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
    break;

  case DISPLAY_MINIMAL_ACTIONS: 
    cout << "Minimal Actions:\n";
    len = 1;
    for(int i = 0; i < side; i++){
      for(int j = 0; j < side; j++){
	stringstream dis;
	if(cell[s].minimalAction == ACTION_NORTH){
	  dis << "^";
	}
	else if(cell[s].minimalAction == ACTION_EAST){
	  dis << ">";
	}
	string str = dis.str();
	for(unsigned int l = 0; l < len; l++){
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
    break;

  case DISPLAY_MAXIMAL_VALUES: 
    cout << "Maximal Values:\n";
    len = 5;
    for(int i = 0; i < side; i++){
      for(int j = 0; j < side; j++){
	stringstream dis;
	dis << cell[s].maximalValue;
	string str = dis.str();
	for(unsigned int l = 0; l < len; l++){
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
    break;

  case DISPLAY_MINIMAL_VALUES: 
    cout << "Minimal Values:\n";
    len = 5;
    for(int i = 0; i < side; i++){
      for(int j = 0; j < side; j++){
	stringstream dis;
	dis << cell[s].minimalValue;
	string str = dis.str();
	for(unsigned int l = 0; l < len; l++){
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
    break;

  default:
    cout << "Incorrect display type.\n";
    return;
  }

}

