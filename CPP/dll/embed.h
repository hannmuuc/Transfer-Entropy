#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
#include <thread>
#include "FFT.h"
#include"entropy.h"
using namespace std;

double calculateMutualInformation(vector<double> X, vector<double> Y);
int claculate_lag(vector<double>& X, int lag_max);
int claculate_lag(vector<double>& X, vector<double>& Y, int lag_max);
vector<vector<double>> embed_vectors(vector<double> data, int lag, int embed);
vector<double> TSE(vector<vector<Complex>>& X, vector<vector<Complex>>& Y, int delay, int k, int l);
vector<double> RTE(vector<vector<Complex>>& X, vector<vector<Complex>>& Y, int delay, int k, int l, double alp);
//多线程处理数据
double transfer_entropy_matirc_item(vector<int>& X, vector<int>& Y, int delay, int k, int l, double Alp);
void transfer_entropy_matirc_thread(vector<vector<int>>& X, int delay, int k, int l, double alp, vector<vector<double>>& res, int start, int end);
vector<vector<double>>  transfer_entropy_matirc(vector<vector<double>>& X, int delay, int k, int l, double alp);