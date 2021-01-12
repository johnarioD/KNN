#ifndef AUXILARY
#define AUXILARY
#include <stdlib.h>
#include <iostream>
#include <cblas.h>
#include <math.h>
#include <bits/stdc++.h>
#include <map>
#include <queue>
#include <vector>
#include <cstdlib>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <cstring>
#include <unordered_map>
using namespace std;

static unordered_map<string, pair<double, int>> medianMap;
static unordered_map<string, vector<double>> vectorMap;
static unordered_map<string, double*> doubleMap;
static unordered_map<string, int*> intMap;
static vector<double> a ;
typedef struct Knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

double* getVector(double* arr, int lineSize , int line){
	//double* result = (double*)malloc(lineSize);
	double* result = new double[lineSize];
	for (int i=line*lineSize; i<(line+1)*lineSize; i++){
		result[i-line*lineSize]=arr[i];
	}
	return result;
}

double* getMatrix(double* arr, int lineSize , int line, int numberOfLines){
	double* result = new double[lineSize*numberOfLines];
	for (int i=line*lineSize; i<(line+numberOfLines)*lineSize; i++){
		result[i-line*lineSize]=arr[i];
	}
	return result;
}
double distance(double* X, double* Y, int size ){

	double number1 = cblas_ddot(size,X,1,X,1);
	double number2 = cblas_ddot(size,Y,1,Y,1);
	double number3 = cblas_ddot(size,X,1,Y,1);

	double result = sqrt(number1 + number2 -2*number3);
	return result;
}




template<typename T>
void printRowMajorArray(T* arr, int x , int y){

	for (int i=0; i<x; i++){
		for (int j=0; j<y; j++){
			cout<<arr[j+i*y]<< " ";
		}
		cout<<""<<endl;
	}
}

void printRowMajorVector(vector<double> arr, int x, int y){

	for (int i=0; i<y; i++){
		for(int j=0; j< x; j++){
			cout<<arr[j+i*x]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
}


vector<double> fromArrayToVector(double* &arr, int dimension){
	vector<double> result(dimension);
	for(int i=0; i< dimension; i++){
		result[i] = arr[i];
	}
	return result;
}

double* fromVectorToArray(const vector<double>& arr){
	double *result = (double*)malloc(arr.size()*sizeof(double));
	for(int i=0; i<arr.size(); i++){
		result[i]=arr[i];
	}
	return result;
}
pair<double, int> findMedian(const vector<double>& X, int lineSize, int dimension, int element){
	priority_queue<double> stuff;
	for(int i=element; i<dimension; i+=lineSize){
		stuff.push(X[i]);
	}
	int numberOfLines = dimension/lineSize;
	int medianIndex = numberOfLines/2;
	int preVal = -1;
	int preCounter = 0;
	for(int i=0; i<medianIndex; i++){

		if(stuff.top()==preVal){
			preCounter++;
		}else{
			preCounter = 0;
		}
		preVal=stuff.top();
		stuff.pop();
	}
	if(stuff.top()==preVal){
		preCounter++;

	}else{
		preCounter=0;
	}

	pair<double, int> result;
	result.first = stuff.top();
	result.second = stuff.size() + preCounter ;
	return result;
}

vector<double> findPoints(const vector<double>& X , int lineSize, int dimension, double median,int amount, int counter, bool greaterThanMedian){


	vector<double> result;

	if(greaterThanMedian){
		for(int i=counter; i<dimension; i+=lineSize){
			if(X[i]>median){
				for(int j=0; j<lineSize; j++){
					result.push_back(X[j+i-counter]);

				}

			}
		}
	}
	else
	{
		for(int i=counter; i<dimension; i+=lineSize){

			if(X[i]<=median){
				for(int j=0; j<lineSize; j++){
					result.push_back(X[j+i-counter]);
				}

			}
		}
	}

	return result;
}


void getVpt(double* data, double* lineY, int lineSize, int dim, int b, string& k){
	int counter=0;
	int numberOfLines = dim/lineSize;


	int amount = numberOfLines;
	if(a.size()==0)
		a = fromArrayToVector(data, dim);

	const vector<double>* X = &a;

	string key="r";

	while(true){

		unordered_map<string,pair<double, int>>::const_iterator medianIt = medianMap.find (key);
		pair<double, int> medianInfo;
		if(medianIt!=medianMap.end()){

			medianInfo = medianIt->second;
		}else{
			medianInfo = findMedian(*(X), lineSize, X->size(), counter);
			medianMap.insert(make_pair(key, medianInfo));
		}

		bool greater = true;
		if(lineY[counter]>medianInfo.first){
			if(amount-medianInfo.second<b)
				break;
			amount = amount-medianInfo.second;
		}
		else
		{
			if(medianInfo.second<b)
				break;
			amount = medianInfo.second;
			greater = false;
		}

		if(greater){
			key += "u";
		}else{
			key+="d";
		}
		unordered_map<string, vector<double>>::const_iterator vectorIt = vectorMap.find(key);
		if(vectorIt!=vectorMap.end()){
			X=&vectorIt->second;
		}else{

			vectorMap.insert(make_pair(key, findPoints(*X, lineSize, X->size(), medianInfo.first, amount, counter, greater)));
			unordered_map<string, vector<double>>::const_iterator vectorIttmp = vectorMap.find(key);

			X = &(vectorIttmp->second);

		}


		counter++;
		counter = counter%lineSize;
	}


	unordered_map<string, double*>::const_iterator doubleIt = doubleMap.find(key);
	if(doubleIt==doubleMap.end()){
		double* tmp = fromVectorToArray(*X);
		doubleMap.insert(make_pair(key, tmp));
	}
	k = key;

}

int* findIndices(double* X, double* Y, int column1, int column2, int lineSize){
	int* result = (int*)malloc(column2*sizeof(int));
	int index=-1;
	for(int i=0; i<column2; i+=lineSize){
		for(int j=0; j<column1; j+=lineSize){
			if(X[j]==Y[i]){
				for(int k=1; k<lineSize; k++){
					if(Y[i+k] != X[j+k]){
						break;
					}
					else if(k==(lineSize-1)&&(Y[i+k] == X[j+k])){
						index = j/lineSize;
					}
				}
			}
		if(index!=-1){
			result[i/lineSize] = index;
			index =-1;
			break;
		}
		}
	}

	return result;
}

pair<double*, double*> readCSV(string name, int m, char delimeter, int& n , int& d){
	vector<double> X;
	vector<double> Y;
	ifstream file(name);
	if(!file.is_open()){
		cout<<"Could not open CSV , exiting program"<<endl;
		exit(1);
	}

	int counterY = 0;
	int D = 0;
	while(file){
		if(counterY==m)
			break;
		string line;
		getline(file,line);
		replace( line.begin(), line.end(), delimeter, ' ');
		istringstream stream (line);
		D=0;
		while(true){

			string tmp;
			stream>>tmp;
			if(tmp==""){
				break;
			}
			D++;
			Y.push_back(atof(tmp.c_str()));
		}
		counterY++;
	}

	while(file){
		string line;
		getline(file,line);
		replace( line.begin(), line.end(), delimeter, ' ');
		istringstream stream (line);
		while(true){
			string tmp;
			stream>>tmp;
			if(tmp==""){
				break;
			}
			X.push_back(atof(tmp.c_str()));
		}
	}
	double* XPointer =fromVectorToArray(X);
	double* YPointer =fromVectorToArray(Y);

	pair<double*, double*> result;
	result.first = XPointer;
	result.second = YPointer;
	n = X.size()/D;
	m = Y.size()/D;
	d = D;

	return result;
}

 vector<string> allVpt(double* X , double* Y, int n, int m, int d, int b){

	vector<string> res1;


	for(int i =0; i< m ; i++){
		double* z = getVector(Y, d, i);
		string key;

		getVpt(X, z, d, n*d, b , key);

		unordered_map<string, int*>::const_iterator intIt = intMap.find(key);
		int* ind;
		if(intIt!=intMap.end()){
			ind = intIt->second;
		}else{

			ind= findIndices(X, doubleMap[key], n*d, vectorMap[key].size(), d);

			intMap.insert(make_pair(key, ind));
		}
		res1.push_back(key);

	}


	return res1;
}

knnresult kNNVpt(double* X, double* Y, int n , int m, int d, int k, vector<string> codes){
	knnresult result;

	const int Ysize = 1;
	int* indices = (int*)malloc(m*k*sizeof(int));
	double* distances =(double*)malloc(m*k*sizeof(double));

	int size = 2*k;

	for(int i=0; i<m ; i++){
		string key = codes[i];
		int x_i = vectorMap[key].size() / d;

		double* XsqVal = (double*)calloc(x_i, sizeof(double));

		for(int j =0; j<x_i; j++){
			for(int d_index=0; d_index<d; d_index++){

				XsqVal[j]+=doubleMap[key][j*d+d_index]*doubleMap[key][j*d+d_index];
			}
		}
		double* Yline = getVector(Y, d , i);
		double YsqVal = 0;
		for(int j=0; j<d; j++){
			YsqVal+=Yline[j]*Yline[j];
		}

		double* tmpDistances = (double*)malloc(x_i*sizeof(double));
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, x_i,Ysize,d, -2.0, doubleMap[key], d, Yline, d, 0.0, tmpDistances, Ysize);

		for(int j=0; j<x_i; j++){
			tmpDistances[j] += YsqVal + XsqVal[j];
		}

		multimap<double, int> tempMap;
		for(int g=0; g<x_i; g++){
			if(g<k){
				tempMap.insert(make_pair(tmpDistances[g], g));
			}
			else{
				multimap<double, int>::iterator it = tempMap.end();
				it--;
				if(it->first > tmpDistances[g]){

					tempMap.erase(it);
					tempMap.insert(make_pair(tmpDistances[g], g));
				}
			}
		}
		int counter = 0;
		for (auto const& x : tempMap)
		{
   			indices[counter+(i)*k] =intMap[key][x.second];
			distances[counter+(i)*k] = x.first;
   			counter++;

		}

		free(tmpDistances);
		free(Yline);

		free(XsqVal);
	}
	result.nidx = indices;
	result.ndist = distances;
	return result;
}
#endif
