#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <cassert>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cmath>

using namespace std;

static const double L_RATE = 2;
static const double pi = 3.14159265359;

class Neuron;
typedef vector<Neuron> Layer;

class Neuron
{
public:
	Neuron();
	Neuron(int prevNum, int index);

	Neuron(int prevNum, int index, double output);

	static double Transfer(double x);
	void FeedForward(Layer& prev);

	void BackPropagateHidden(Layer& next, Layer& prev);
	void BackPropagateLast(double expected, Layer& prev);

public:
	double output;
	vector<double> weights;
	double e;
	int index;
};

class Network
{
public:
	Network(string fmt);

	void Initiate(vector<vector<double> >& input, vector<vector<double> >& expected);

	void Train();

	void SaveWeights(string dir);
	void LoadWeights(string dir);

	vector<double> Feed(vector<double> input);

	vector<Layer> layers;
	vector<vector<double> > inputs;
	vector<vector<double> > expected;

	double error;
};

#endif