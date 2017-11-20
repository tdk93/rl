#include "MDP.h"

MDP::MDP(double strideLength, double stateNoiseSigma, double actionNoiseSigma){

  this->strideLength = strideLength;
  this->stateNoiseSigma = stateNoiseSigma;
  this->actionNoiseSigma = actionNoiseSigma;

  srand48(time(0));

  fixTerminal();
  reset();

  lastReward = 0;
}

MDP::~MDP(){
}

double MDP::normal(const double &mean, const double &std){

  double u1, u2, v1, v2;

  double s = 2.0;

  while(s >= 1.0){

  u1 = drand48();
  v1 = 2.0 * u1 - 1.0;

  u2 = drand48();
  v2 = 2.0 * u2 - 1.0;

  s = v1 * v1 + v2 * v2;

  }
  
  double x1 = v1 * sqrt(-2.0 * log(s) / s);
  double value = (x1 * std) + mean;

  return value;
}

void MDP::fixTerminal(){

  terminalSquareSide = sqrt(TERMINI_FRACTION * 1.0 / NUM_TERMINAL_SQUARES);

  for(int i = 0; i < NUM_TERMINAL_SQUARES; i++){
    
    bool collide;
    do{
      
      collide = false;

      double x = (drand48() - 0.5) * (1.0 - terminalSquareSide);
      double y = (drand48() - 0.5) * (1.0 - terminalSquareSide);

      terminalSquareCentres[i] = VecPosition(x, y);

      for(int j = 0; j < i; j++){

	VecPosition topLeft = VecPosition(-terminalSquareSide * 0.5, terminalSquareSide * 0.5);
	VecPosition topRight = VecPosition(terminalSquareSide * 0.5, terminalSquareSide * 0.5);
	VecPosition bottomLeft = VecPosition(-terminalSquareSide * 0.5, -terminalSquareSide * 0.5);
	VecPosition bottomRight = VecPosition(terminalSquareSide * 0.5, -terminalSquareSide * 0.5);

	if(isTerminal(terminalSquareCentres[i] + topLeft, j)){
	  collide = true;
	}
	else if(isTerminal(terminalSquareCentres[i] + topRight, j)){
	  collide = true;
	}
	else if(isTerminal(terminalSquareCentres[i] + bottomLeft, j)){
	  collide = true;
	}
	else if(isTerminal(terminalSquareCentres[i] + bottomRight, j)){
	  collide = true;
	}

      }

    }
    while(collide);

  }
  
}

void MDP::reset(){

  do{

    double x = drand48() - 0.5;
    double y = drand48() - 0.5;

    currentPosition = VecPosition(x, y);
  }
  while(isTerminal(currentPosition));

}

void MDP::getState(double state[]){
  
  double x, y;
  
  if(stateNoiseSigma == 0){
    
    x = currentPosition.getX();
    y = currentPosition.getY();
  }
  else{

    bool valid = true;

    do{

      x = normal(currentPosition.getX(), stateNoiseSigma);
      y = normal(currentPosition.getY(), stateNoiseSigma);

      valid = (x < 0.5) && (x > -0.5) && (y < 0.5) && (y > -0.5);
    }
    while(!valid);
  }

  int j = 0;
  state[j++] = x;
  state[j++] = y;
}

bool MDP::isTerminal(VecPosition point, int terminalSquareIndex){

  VecPosition diff = terminalSquareCentres[terminalSquareIndex] - point;
  if((fabs(diff.getX()) < (terminalSquareSide * 0.5)) && (fabs(diff.getY()) < (terminalSquareSide * 0.5))){
    return true;
  }

  return false;
}

bool MDP::isTerminal(VecPosition point){

  for(int i = 0; i < NUM_TERMINAL_SQUARES; i++){
    if(isTerminal(point, i)){
      return true;
    }
  }

  return false;
}

bool MDP::takeAction(int action){

  VecPosition stride;

  if(action == ACTION_NORTH){
    stride = VecPosition(0, strideLength);
  }
  else if(action == ACTION_SOUTH){
    stride = VecPosition(0, -strideLength);
  }
  else if(action == ACTION_EAST){
    stride = VecPosition(strideLength, 0);
  }
  else{// if(action == ACTION_WEST)
    stride = VecPosition(-strideLength, 0);
  }

  if(actionNoiseSigma != 0){
    double angle = normal(0, actionNoiseSigma);
    stride.rotate(angle);
  }

  currentPosition = currentPosition + stride;

  /*
  if(currentPosition.getX() < -0.5){
    currentPosition.setX(-0.5);
  }
  else if(currentPosition.getX() > 0.5){
    currentPosition.setX(0.5);
  }

  if(currentPosition.getY() < -0.5){
    currentPosition.setY(-0.5);
  }
  else if(currentPosition.getY() > 0.5){
    currentPosition.setY(0.5);
  }
  */

  bool terminal = isTerminal(currentPosition);
  bool boundary = false;

  if(currentPosition.getX() < -0.5){
    boundary = true;
  }
  else if(currentPosition.getX() > 0.5){
    boundary = true;
  }

  if(currentPosition.getY() < -0.5){
    boundary = true;
  }
  else if(currentPosition.getY() > 0.5){
    boundary = true;
  }

  if(terminal){

    lastReward = 100.0;
    reset();
  }
  else if(boundary){

    lastReward = 0;
    reset();
  }
  else{
    
    lastReward = -1.0;
  }

  return (terminal || boundary);
}


void MDP::display(){

  
  double increment = 0.025;

  double state[2];
  getState(state);

  for(double y = 0.5; y >= -0.5; y -= 0.025){
    
    for(double x = -0.5; x <= 0.5; x += 0.025){
      
      VecPosition point = VecPosition(x, y);
      VecPosition agentVector = point - currentPosition;

      VecPosition stateVector = point - VecPosition(state[0], state[1]);

      bool agent = (fabs(agentVector.getX()) < (increment * 0.5)) && (fabs(agentVector.getY()) < (increment * 0.5));
      bool percept = (fabs(stateVector.getX()) < (increment * 0.5)) && (fabs(stateVector.getY()) < (increment * 0.5));


      if(isTerminal(point)){
	cout << "T";
      }
      else if(agent){
	cout << "a";
      }
      else if(percept){
	cout << "p";
      }
      else{
	cout << ".";
      }
    }
    
    cout << "\n";
  }

}

double MDP::getLastReward(){

  return lastReward;
}

