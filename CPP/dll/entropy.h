#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <unordered_map>
#include "FFT.h"

using namespace std;

//打印
void print_vector(vector<double>& a);
void print_vector(vector<int>& a);
void print_vector(vector<pair<int, int>>& a);
//打印矩阵
void print_matru(vector<vector<int>>& a);
void print_matru(vector<vector<double>>& a);
void print_matru(vector<vector<pair<int, int>>>& a);
void print_matru(std::vector<std::vector<Complex>>& a);
//计算gauus值
double gauss(double x, double sigma, double mu);
vector<double> gauss_mul(vector<double>& x, vector<double>& y, vector<double>& h, int k, int l);
vector<double> gauss_sum(vector<double> x, vector<vector<double>>& y, vector<double>& h, int k, int l);
//归一化处理 并且生成直方图
vector<int> prework(vector<double>& X);
vector<pair<int, int>> prework(vector<Complex>& X);
//生成一个矩阵
vector<vector<int>> get_matru(vector<int>& x, vector<int>& y, int delay, int k, int l);
vector<vector<pair<int, int>>> get_matru(vector<pair<int, int>>& x, vector<pair<int, int>>& y, int delay, int k, int l);
void get_matru(vector<int>& x, vector<int>& y, int delay, int k, int l, vector<vector<int>>* res);
//数据X Y delay时间延迟 k l 大小
double transfer_entropy(vector<double> X, vector<double> Y, int delay, int k, int l);
//广义熵
double trantransfer_Renyi_entropy(vector<double> X, vector<double> Y, int delay, int k, int l, double Alp);
//复数计算转移熵
double trantransfer_entropy_Complex(vector<Complex> X, vector<Complex> Y, int delay, int k, int l);

//计算广义熵
double trantransfer_Renyi_entropy_Complex(vector<Complex> X, vector<Complex> Y, int delay, int k, int l, double Alp);