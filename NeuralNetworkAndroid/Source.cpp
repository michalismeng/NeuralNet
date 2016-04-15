#include "NeuralNetwork.h"
#include <iostream>

using namespace std;

// This is a new commit

void Print(vector<double> vec)
{
	for (int i = 0; i < vec.size(); i++)
		cout << vec[i] << " ";

	cout << endl;
}

extern "C" void* Initialize()
{
	Network* net = new Network("2 5 1");

	vector<vector<double> > inputs;
	vector<double> temp;
	temp.push_back(0);
	temp.push_back(0);
	inputs.push_back(temp);

	temp[0] = 1;
	temp[1] = 1;
	inputs.push_back(temp);

	vector<vector<double> > outputs;
	vector<double> temp2;
	temp2.push_back(1);
	outputs.push_back(temp2);

	temp2[0] = 0;
	outputs.push_back(temp2);

	cout << "vector ready" << endl;

	//net.LoadWeights("TestXOR.txt");

	net->Initiate(inputs, outputs);

	cout << "Network initialized" << endl;

	for (int i = 0; net->error > 0.1; i++)
	{
		if (i % 1000 == 0)
			cout << net->error << endl;

		net->Train();
	}

	return net;
}

extern "C" void Feed(void* network, double* params)
{
	Network& net = *(Network*)network;

	vector<double> temp;

	for (int i = 0; i < 2; i++)
		temp.push_back(params[i]);

	Print(net.Feed(temp));

	cout << "End print" << endl;
}
