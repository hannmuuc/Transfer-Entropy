#include "pch.h"
#include"embed.h"


//互信息法确定lag 等隔间距法
double calculateMutualInformation(vector<double> X, vector<double> Y) {
	if (X.size() != Y.size())return 0;
	vector<int> X_prework;
	vector<int> Y_prework;
	X_prework = prework(X);
	Y_prework = prework(Y);

	map<pair<int, int>, int> psq;
	map<int, int> ps;
	map<int, int> pq;

	for (int i = 0;i < X_prework.size();i++) {
		int a = X_prework[i], b = Y_prework[Y_prework.size() - 1 - i];
		pair<int, int> c = { a,b };
		psq[c]++;
		ps[a]++;
		pq[b]++;
	}
	double num = X_prework.size();
	double mI = 0;
	for (auto it = psq.begin();it != psq.end();it++) {
		double psq_item = 1.0 * it->second / num;
		int a = it->first.first, b = it->first.second;
		double ps_item = 1.0 * ps[a] / num;
		double pq_item = 1.0 * pq[b] / num;

		if (psq_item == 0)mI += 0;
		else mI += psq_item * log2(psq_item / (ps_item * pq_item));
	}
	return mI;
}
//计算出X和Y最不相关的lag
int claculate_lag(vector<double>& X, int lag_max) {
	if (X.size() <= 0)
		return -1;
	vector<double> X_pre;
	vector<double> X_end;
	for (int i = 1;i < X.size();i++) {
		X_pre.push_back(X[i]);
		X_end.push_back(X.size() - 1 - i);
	}
	if (X_pre.size() <= 0)return -1;


	double mutualinformation = calculateMutualInformation(X_pre, X_end);
	int lag = 1;
	int lag_num = 1;
	X_pre.pop_back();
	X_end.pop_back();
	while (X_pre.size() > 0)
	{
		lag_num++;
		double information = calculateMutualInformation(X_pre, X_end);

		if (information < mutualinformation) {
			mutualinformation = information;
			lag = lag_num;
		}
		else {
			break;
		}
		X_pre.pop_back();
		X_end.pop_back();

	}

	if (lag_max >= 1) {
		lag = min(lag, lag_max);
	}

	return lag;

}

//计算出X和Y最不相关的lag X和Y互信息的平均值
int claculate_lag(vector<double>& X, vector<double>& Y, int lag_max) {
	if (X.size() != Y.size())
		return -1;
	vector<double> X_pre;
	vector<double> X_end;
	vector<double> Y_pre;
	vector<double> Y_end;
	for (int i = 1;i < X.size();i++) {
		X_pre.push_back(X[i]);
		X_end.push_back(X.size() - 1 - i);
		Y_pre.push_back(Y[i]);
		Y_end.push_back(Y.size() - 1 - i);
	}
	if (X_pre.size() <= 0)return -1;
	if (Y_pre.size() <= 0)return -1;

	double mutualinformation_x = calculateMutualInformation(X_pre, X_end);
	double mutualinformation_y = calculateMutualInformation(Y_pre, Y_end);
	double mutualinformation = mutualinformation_x + mutualinformation_y;
	int lag = 1;
	int lag_num = 1;
	X_pre.pop_back();
	X_end.pop_back();
	Y_pre.pop_back();
	Y_end.pop_back();

	while (X_pre.size() > 0)
	{
		lag_num++;
		if (lag_num > lag_max)return lag;
		double information_x = calculateMutualInformation(X_pre, X_end);
		double information_y = calculateMutualInformation(Y_pre, Y_end);
		double information = information_x + information_y;
		if (information < mutualinformation) {
			mutualinformation = information;
			lag = lag_num;
		}
		else {
			break;
		}

		X_pre.pop_back();
		X_end.pop_back();
		Y_pre.pop_back();
		Y_end.pop_back();
	}

	if (lag_max >= 1) {
		lag = min(lag, lag_max);
	}

	return lag;

}

//相空间变换 embed 为宽 
vector<vector<double>> embed_vectors(vector<double> data, int lag, int embed) {
	vector<vector<double>> embd;
	for (int i = 0;i + (embed - 1) * lag < data.size();i++) {
		embd.push_back({});
		for (int j = 1;j <= embed;j++) {
			embd[i].push_back(data[i + (j - 1) * lag]);
		}
	}
	return embd;

}


//计算所有频率的TSE
vector<double> TSE(vector<vector<Complex>>& X, vector<vector<Complex>>& Y, int delay, int k, int l) {
	vector<double> TSE_res;
	if (X.size() != Y.size())return TSE_res;
	int len = X.size();
	//使用一半
	for (int i = 0;i < len / 2;i++) {
		double TE_f = trantransfer_entropy_Complex(X[i], Y[i], delay, k, l);
		TSE_res.push_back(TE_f);
	}
	return TSE_res;
}

//计算所有频率的RTE
vector<double> RTE(vector<vector<Complex>>& X, vector<vector<Complex>>& Y, int delay, int k, int l, double alp) {
	vector<double> TSE_res;
	if (X.size() != Y.size())return TSE_res;
	int len = X.size();
	//使用一半
	for (int i = 0;i < len / 2;i++) {
		double TE_f = trantransfer_Renyi_entropy_Complex(X[i], Y[i], delay, k, l, alp);
		TSE_res.push_back(TE_f);
	}
	return TSE_res;
}

double transfer_entropy_matirc_item(vector<int>& X, vector<int>& Y, int delay, int k, int l, double Alp) {

	vector<vector<int>>* b = new vector<vector<int>>;

	get_matru(X, Y, delay, k, l, b);
	vector<vector<int>>& a = (*b);

	//auto a = get_matru(X, Y, delay, k, l);

	int length = 5 * X.size() + 1;
	vector<int> bin_1(length, 0);
	vector<int> bin_2(length, 0);
	vector<int> bin_3(length, 0);
	vector<int> bin_4(length, 0);
	vector<unsigned long long> base;
	base.push_back(1);
	unsigned long long base_flag = 1;
	const unsigned long long base_num = 1e9 + 7;

	int op_sum = (1 + k + l);
	for (int i = 1;i <= op_sum;i++) {
		base_flag = base_flag * base_num;
		base.push_back(base_flag);
	}

	int data_sum = a.size();
	for (int i = 0;i < a.size();i++) {
		unsigned long long bin_1_tool = 0, bin_2_tool = 0, bin_3_tool = 0, bin_4_tool = 0;
		unsigned long long bin_base = a[i][0];
		bin_1_tool += bin_base;
		bin_3_tool += bin_base;
		for (int j = 1;j <= k;j++) {
			unsigned long long bin_base = base[j] * a[i][j];
			bin_1_tool += bin_base;
			bin_2_tool += bin_base;
			bin_3_tool += bin_base;
			bin_4_tool += bin_base;
		}
		for (int j = k + 1;j < a[i].size();j++) {
			unsigned long long bin_base = base[j] * a[i][j];
			bin_1_tool += bin_base;
			bin_4_tool += bin_base;
		}
		int tool1 = bin_1_tool % length;
		int tool2 = bin_2_tool % length;
		int tool3 = bin_3_tool % length;
		int tool4 = bin_4_tool % length;
		bin_1[tool1]++;
		bin_2[tool2]++;
		bin_3[tool3]++;
		bin_4[tool4]++;
	}

	double TE = 0;
	//bin_1_p pyyx2  bin_2_p py bin_3_p pxy bin_4_p pyx2
	double bin_1_p = 0, bin_2_p = 0, bin_3_p = 0, bin_4_p = 0;
	for (auto it = bin_1.begin();it != bin_1.end();it++) {
		double p = 1.0 * (*it) / data_sum;
		bin_1_p += pow(p, Alp);
	}
	for (auto it = bin_2.begin();it != bin_2.end();it++) {
		double p = 1.0 * (*it) / data_sum;
		bin_2_p += pow(p, Alp);
	}
	for (auto it = bin_3.begin();it != bin_3.end();it++) {
		double p = 1.0 * (*it) / data_sum;
		bin_3_p += pow(p, Alp);
	}
	for (auto it = bin_4.begin();it != bin_4.end();it++) {
		double p = 1.0 * (*it) / data_sum;
		bin_4_p += pow(p, Alp);
	}
	TE = log2(bin_3_p * bin_4_p / (bin_1_p * bin_2_p)) / (1 - Alp);

	return TE;
}

void transfer_entropy_matirc_thread(vector<vector<int>>& X, int delay, int k, int l, double alp, vector<vector<double>>& res, int start, int end) {
	for (int i = start;i < end;i++) {
		for (int j = 0;j < X.size();j++) {
			if (i == j) { res[i][j] = 0; continue; }
			else {
				res[i][j] = transfer_entropy_matirc_item(X[i], X[j], delay, k, l, alp);
			}
		}
		cout << "row:" << i << " finshed\n";
	}
	return;
}

vector<vector<double>>  transfer_entropy_matirc(vector<vector<double>>& X, int delay, int k, int l, double alp) {
	int length = X.size();
	vector<vector<double>> res(length, vector<double>(length));
	auto Y = new vector<vector<int>>;
	//vector<vector<int>> Y;
	for (int i = 0;i < length;i++) {
		Y->push_back(prework(X[i]));
	}
	//transfer_entropy_matirc_thread((*Y), delay, k, l, alp, res, 0, length);


	thread th[10];
	int thread_num = min(2, length);
	int unit = length / thread_num;
	if (unit * thread_num != length)
		unit++;
	cout << "thread:" << thread_num << endl;
	for (int i = 0; i < thread_num; i++) {
		int start = i * unit;
		int end = min((i + 1) * unit, length);
		th[i] = thread(transfer_entropy_matirc_thread, ref(*Y), delay, k, l, alp, ref(res), start, end);
	}

	for (int i = 0;i < thread_num;i++) {
		th[i].join();
	}
	delete Y;
	return res;
}
