// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include "embed.h"
#include "entropy.h"
#include "FFT.h"


//数组 X Y 数组大小 num  时延delay  k l
double TE(double* X, double* Y, int num, int delay, int k, int l) {
	vector<double> x, y;
	for (int i = 0;i < num;i++) {
		x.push_back(X[i]);
		y.push_back(Y[i]);
	}
	double TE1;
	TE1 = transfer_entropy(x, y, delay, k, l);
	return TE1;
}

double RTE(double* X, double* Y, int num, int delay, int k, int l, double alp) {
	vector<double> x, y;
	for (int i = 0;i < num;i++) {
		x.push_back(X[i]);
		y.push_back(Y[i]);
	}
	double TE1;
	TE1 = trantransfer_Renyi_entropy(x, y, delay, k, l, alp);
	return TE1;

}

Node_pointer TSE(double* X, double* Y, int num, int delay, int k, int l) {
	vector<double> x, y;
	for (int i = 0;i < num;i++) {
		x.push_back(X[i]);
		y.push_back(Y[i]);
	}

	//使用参数为8
	int lag = claculate_lag(x, y, 4);
	auto a = embed_vectors(x, lag, 8);
	auto c = embed_vectors(y, lag, 8);

	//进行FFt变换
	auto b = FFT_2D(a);
	auto d = FFT_2D(c);

	auto tse = TSE(b, d, delay, k, l);

	Node_pointer data_tse = (Node_pointer)malloc(sizeof(node));
	data_tse->data_num = 0;

	int ave = (int)tse.size() / 16;

	num = 0;
	double sum = 0;
	for (int i = 0;i < tse.size();i++) {
		sum += tse[i];
		num++;
		if (num >= ave) {
			sum /= num;
			data_tse->data[data_tse->data_num] = sum;
			data_tse->data_num++;
			sum = 0;
			num = 0;
		}
	}
	return data_tse;
}

Node_pointer RTSE(double* X, double* Y, int num, int delay, int k, int l, double alp) {
	vector<double> x, y;
	for (int i = 0;i < num;i++) {
		x.push_back(X[i]);
		y.push_back(Y[i]);
	}

	//使用参数为8
	int lag = claculate_lag(x, y, 4);
	auto a = embed_vectors(x, lag, 8);
	auto c = embed_vectors(y, lag, 8);

	//进行FFt变换
	auto b = FFT_2D(a);
	auto d = FFT_2D(c);
	auto tse = RTE(b, d, delay, k, l, alp);
	Node_pointer data_tse = (Node_pointer)malloc(sizeof(node));
	data_tse->data_num = 0;

	int ave = (int)tse.size() / 16;

	num = 0;
	double sum = 0;
	for (int i = 0;i < tse.size();i++) {
		sum += tse[i];
		num++;
		if (num >= ave) {
			sum /= num;
			data_tse->data[data_tse->data_num] = sum;
			data_tse->data_num++;
			sum = 0;
			num = 0;
		}
	}
	return data_tse;
}

Node_pointer TE_matric(double* X, int cow, int row, int delay, int k, int l, double alp) {
	//构件矩阵
	auto x = new vector<vector<double>>;
	for (int i = 0;i < cow;i++) {
		x->push_back({});
		for (int j = 0;j < row;j++) {
			(*x)[i].push_back(X[i * row + j]);
		}
	}
	Node_pointer data_tse = new node;

	data_tse->data_num = 0;
	auto res = transfer_entropy_matirc((*x), delay, k, l, alp);

	delete x;

	for (int i = 0;i < res.size();i++) {
		for (int j = 0;j < res.size();j++) {
			data_tse->data[data_tse->data_num] = res[i][j];
			data_tse->data_num++;
		}
	}
	return data_tse;
}
