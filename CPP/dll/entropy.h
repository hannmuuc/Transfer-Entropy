#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <unordered_map>
#include "FFT.h"

using namespace std;

//��ӡ
void print_vector(vector<double>& a);
void print_vector(vector<int>& a);
void print_vector(vector<pair<int, int>>& a);
//��ӡ����
void print_matru(vector<vector<int>>& a);
void print_matru(vector<vector<double>>& a);
void print_matru(vector<vector<pair<int, int>>>& a);
void print_matru(std::vector<std::vector<Complex>>& a);
//����gauusֵ
double gauss(double x, double sigma, double mu);
vector<double> gauss_mul(vector<double>& x, vector<double>& y, vector<double>& h, int k, int l);
vector<double> gauss_sum(vector<double> x, vector<vector<double>>& y, vector<double>& h, int k, int l);
//��һ������ ��������ֱ��ͼ
vector<int> prework(vector<double>& X);
vector<pair<int, int>> prework(vector<Complex>& X);
//����һ������
vector<vector<int>> get_matru(vector<int>& x, vector<int>& y, int delay, int k, int l);
vector<vector<pair<int, int>>> get_matru(vector<pair<int, int>>& x, vector<pair<int, int>>& y, int delay, int k, int l);
void get_matru(vector<int>& x, vector<int>& y, int delay, int k, int l, vector<vector<int>>* res);
//����X Y delayʱ���ӳ� k l ��С
double transfer_entropy(vector<double> X, vector<double> Y, int delay, int k, int l);
//������
double trantransfer_Renyi_entropy(vector<double> X, vector<double> Y, int delay, int k, int l, double Alp);
//��������ת����
double trantransfer_entropy_Complex(vector<Complex> X, vector<Complex> Y, int delay, int k, int l);

//���������
double trantransfer_Renyi_entropy_Complex(vector<Complex> X, vector<Complex> Y, int delay, int k, int l, double Alp);