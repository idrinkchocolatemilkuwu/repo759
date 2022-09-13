#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
  int N;
  N = atoi(argv[1]);
	for (int i = 0; i < N+1; i++) {
		printf("%i ",i);
	}
	cout << "\n";
	for (int i = N; i > -1; i--) {
		cout << i << " ";
	}
  return 0;
}
