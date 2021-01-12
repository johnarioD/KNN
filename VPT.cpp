#include <iostream>
#include <bits/stdc++.h>
#include <map>
#include <time.h>
#include "auxiliary.hpp"

using namespace std;


int main(){
	struct timespec ts_start, ts_end;
	//test data
	double X[200]={38,176,43,163,38,131,40,133,49,119,46,142,33,142,40,180,28,183,31,132,45,128,42,137,25,174,39,202,36,129,48,181,32,191,27,131,37,179,50,172,48,133,39,117,41,137,44,146,28,123,25,189,39,143,25,114,36,166,30,186,45,126,40,137,25,138,47,187,44,193,48,137,44,192,35,118,33,180,38,128,39,164,44,183,44,169,37,194,45,172,37,135,30,182,39,121,42,158,42,179,49,170,44,136,43,135,47,147,50,186,38,124,41,134,45,170,36,180,38,130,29,130,28,127,30,141,28,111,29,134,36,189,45,137,32,136,31,130,48,137,25,186,40,127,39,176,41,127,33,115,31,178,35,131,32,183,42,194,48,126,34,186,39,188,28,189,29,120,32,132,39,182,37,120,49,123,31,141,37,129,38,184,45,181,30,124,48,174,48,134,25,171,44,188,49,186,45,172,48,177};
	double Y[10]={20,162,30,169,40,168,50,170,60,171};
	double B[4]={3.0,0.2,0.0,5.0};
	double C[4]={0.0,0.0,0.0,0.0};
	
	//end of test data
	int m=1;
	int n , d;
	pair<double*, double*> result = readCSV("./Meas/FMA/features.csv",m,',',n,d); // Read input file from here
	n = 1250;

	vector<string> ress = allVpt(result.first,result.first , n, n, d, 1000);

	clock_gettime(CLOCK_MONOTONIC, &ts_start);
	knnresult res=kNNVpt(result.first, result.first, n,n,d,15,ress);
	clock_gettime(CLOCK_MONOTONIC, &ts_end);

	printf("Time: %ld.%ld\n", (ts_end.tv_sec - ts_start.tv_sec), (abs(ts_end.tv_nsec - ts_start.tv_nsec)));
	return 0;
}
