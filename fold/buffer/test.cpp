#include <time.h>
#include <iostream>

using namespace std;

int main(){

  srand48(time(0));

  for(int i = 0; i < 255; i++){
    cout << (char)(i) << "\n";
  }

  return 0;
}
