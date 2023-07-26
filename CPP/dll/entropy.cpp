#include "pch.h"
#include"entropy.h"


//打印
void print_vector(vector<double>& a) {
	for (int i = 0;i < a.size();i++) {
		cout << a[i] << " ";
	}cout << endl;
}

void print_vector(vector<int>& a) {
	for (int i = 0;i < a.size();i++) {
		cout << a[i] << " ";
	}cout << endl;
}

void print_vector(vector<pair<int, int>>& a) {
	for (int i = 0;i < a.size();i++) {
		cout << a[i].first << "/" << a[i].second << " ";
	}cout << endl;
}

//计算double_matru
void print_matru(vector<vector<double>>& a) {
	for (int i = 0;i < a.size();i++) {
		for (int j = 0;j < a[i].size();j++) {
			cout << a[i][j] << " ";
		}cout << endl;
	}
	return;
}

void print_matru(vector<vector<int>>& a) {
	for (int i = 0;i < a.size();i++) {
		for (int j = 0;j < a[i].size();j++) {
			cout << a[i][j] << " ";
		}cout << endl;
	}
	return;
}

void print_matru(vector<vector<pair<int, int>>>& a) {
	for (int i = 0;i < a.size();i++) {
		for (int j = 0;j < a[i].size();j++) {
			cout << a[i][j].first << "/" << a[i][j].second << " ";
		}cout << endl;
	}
	return;
}

void print_matru(std::vector<std::vector<Complex>>& a) {
	for (int i = 0;i < a.size();i++) {
		for (int j = 0;j < a[i].size();j++) {
			std::cout << a[i][j] << " ";
		}std::cout << std::endl;
	}
	return;
}
//gauss函数 x mu为中间值 sigma为参数
double gauss(double x, double sigma, double mu) {
	double coefficient = 1 / (sigma * std::sqrt(2 * 3.1415926));
	double exponent = -0.5 * std::pow((x - mu) / sigma, 2);
	return coefficient * std::exp(exponent);
}

// x y h为相同维度的向量
vector<double> gauss_mul(vector<double>& x, vector<double>& y, vector<double>& h, int k, int l) {
	int length = 1 + k + l;
	vector<double> res(4);
	//不为向量无法计算
	if (length != x.size() || length != y.size() || length != h.size()) {
		res[0] = -1;
		return res;
	}
	//cout << "----------------\n";
	//print_vector(x);
	//print_vector(y);
	//print_vector(h);
	//cout << "-------------------\n";

	double pxpkpl = 1;
	double pk = 1;
	double pkpl = 1;
	double pxpk = 1;
	double pl = 1;
	double px = 1;

	double a = (x[0] - y[0]) / h[0];
	if (a > 1 || a < -1) {
		//pk = 0;
		px = px * gauss(a, 1, 0);
	}
	else {
		px = px * gauss(a, 1, 0);//计算gauss值
	}

	for (int i = 1;i <= k;i++) {
		double a = (x[i] - y[i]) / h[i];
		/*
		if (a > 1 || a < -1) {
			pk = 0;
			break;
		}*/
		pk = pk * gauss(a, 1, 0);//计算gauss值
	}
	for (int i = k + 1;i < length;i++) {
		double a = (x[i] - y[i]) / h[i];
		/*
		if (a > 1 || a < -1) {
			pl = 0;
			break;
		}*/
		pl = pl * gauss(a, 1, 0);//计算gauss值

	}
	pkpl = pk * pl;
	pxpk = px * pk;
	pxpkpl = px * pkpl;
	res = { pxpkpl,pk,pkpl,pxpk };

	return res;
}
// 核函数计算 x 是输入的值
vector<double> gauss_sum(vector<double> x, vector<vector<double>>& y, vector<double>& h, int k, int l) {
	int length = y.size();
	vector<double> res(4);

	double h_mul_px = h[0];
	double h_mul_pk = 1;
	double h_mul_pl = 1;
	for (int i = 1;i <= k;i++) {
		h_mul_pk *= h[i];

	}
	for (int i = k + 1;i < 1 + k + l;i++) {
		h_mul_pl *= h[i];
	}

	for (int i = 0;i < length;i++) {
		vector<double> a = gauss_mul(x, y[i], h, k, l);
		if (a[0] == -1) {
			res[0] = -1;
			return res;
		}
		for (int j = 0;j < 4;j++)
			res[j] += a[j];
	}
	//防止为0
	res[0] = res[0] / (h_mul_px * h_mul_pk * h_mul_pl * length) + 1e-18;
	res[1] = res[1] / (h_mul_pk * length) + 1e-18;
	res[2] = res[2] / (h_mul_pk * h_mul_pl * length) + 1e-18;
	res[3] = res[3] / (h_mul_px * h_mul_pk * length) + 1e-18;

	//cout << "he:";
	//print_vector(res);
	//cout << res[0] * res[1] - res[2] * res[3] << endl;
	return res;
}

//归一化处理 并且生成直方图
vector<int> prework(vector<double>& X) {
	int bins = (int)sqrt(X.size());
	double X_max = X[0], X_min = X[0];
	for (int i = 1;i < X.size();i++) {
		X_max = max(X_max, X[i]);
		X_min = min(X_min, X[i]);
	}
	double unit;
	if (X_max == X_min)unit = X_max;
	else unit = (X_max - X_min) / bins;
	vector<int> res;
	for (int i = 0;i < X.size();i++) {
		int a = (X[i] - X_min) / unit;
		if (a == bins)a--;
		res.push_back(a + 1);
	}
	return res;
}


vector<pair<int, int>> prework(vector<Complex>& X) {
	vector<pair<int, int>> res;
	if (X.size() == 0)return res;

	int bins = (int)sqrt(X.size());
	double real_min = X[0].real, real_max = X[0].real;
	double imagin_min = X[0].imagin, imagin_max = X[0].imagin;
	for (int i = 1;i < X.size();i++) {
		real_min = min(real_min, X[i].real);
		real_max = max(real_max, X[i].real);
		imagin_min = min(imagin_min, X[i].imagin);
		imagin_max = max(imagin_max, X[i].imagin);
	}
	double unit_real, unit_imagin;
	if (real_max == real_min)unit_real = real_max;
	else unit_real = (real_max - real_min) / bins;
	if (imagin_max == imagin_min)unit_imagin = imagin_min;
	else unit_imagin = (imagin_max - imagin_min) / bins;

	for (int i = 0;i < X.size();i++) {
		int a = (X[i].real - real_min) / unit_real;
		int b = (X[i].imagin - imagin_min) / unit_imagin;
		if (a == bins)a--;
		if (b == bins)b--;
		res.push_back({ a + 1,b + 1 });
	}
	return res;
}

//生成一个矩阵
vector<vector<int>> get_matru(vector<int>& x, vector<int>& y, int delay, int k, int l) {
	vector<vector<int>> res;
	int data_num = min(x.size(), y.size());
	int num = max(k, l);
	for (int i = num;i + delay - 1 < data_num;i++) {
		res.push_back({});
		res[i - num].push_back(x[i + delay - 1]);

		for (int j = i - k;j < i;j++) {
			res[i - num].push_back(x[j]);
		}
		for (int j = i - l;j < i;j++) {
			res[i - num].push_back(y[j]);
		}
	}
	return res;
}

//生成一个矩阵
void get_matru(vector<int>& x, vector<int>& y, int delay, int k, int l, vector<vector<int>>* res) {
	int data_num = min(x.size(), y.size());
	int num = max(k, l);
	for (int i = num;i + delay - 1 < data_num;i++) {
		(*res).push_back({});
		(*res)[i - num].push_back(x[i + delay - 1]);

		for (int j = i - k;j < i;j++) {
			(*res)[i - num].push_back(x[j]);
		}
		for (int j = i - l;j < i;j++) {
			(*res)[i - num].push_back(y[j]);
		}
	}
	return;
}

vector<vector<pair<int, int>>> get_matru(vector<pair<int, int>>& x, vector<pair<int, int>>& y, int delay, int k, int l) {
	vector<vector<pair<int, int>>> res;
	int data_num = min(x.size(), y.size());
	int num = max(k, l);
	for (int i = num;i + delay - 1 < data_num;i++) {
		res.push_back({});
		res[i - num].push_back(x[i + delay - 1]);

		for (int j = i - k;j < i;j++) {
			res[i - num].push_back(x[j]);
		}
		for (int j = i - l;j < i;j++) {
			res[i - num].push_back(y[j]);
		}
	}
	return res;
}

//计算转移熵

//数据X Y delay时间延迟 k l 大小
double transfer_entropy(vector<double> X, vector<double> Y, int delay, int k, int l) {
	if (X.size() != Y.size())return 0;
	vector<int> X_prework;
	vector<int> Y_prework;
	X_prework = prework(X);
	Y_prework = prework(Y);

	auto a = get_matru(X_prework, Y_prework, delay, k, l);
	//print_matru(a);

	map<vector<unsigned long long>, int> bin_1;
	map<unsigned long long, int> bin_2;
	map<unsigned long long, int> bin_3;
	map<unsigned long long, int> bin_4;
	vector<unsigned long long> base;
	base.push_back(1);
	unsigned long long base_flag = 1;
	const unsigned long long base_num = 1e9 + 7;
	for (int i = 1;i <= 1 + k + l;i++) {
		base_flag = base_flag * base_num;
		base.push_back(base_flag);
	}

	int data_sum = a.size();
	for (int i = 0;i < a.size();i++) {
		vector<unsigned long long> sum;
		unsigned long long bin_1_tool = 0, bin_2_tool = 0, bin_3_tool = 0, bin_4_tool = 0;
		bin_1_tool += a[i][0];
		bin_3_tool += a[i][0];
		for (int j = 1;j <= k;j++) {
			unsigned long long bin_base = a[i][j] * base[j];
			bin_1_tool += bin_base;
			bin_2_tool += bin_base;
			bin_3_tool += bin_base;
			bin_4_tool += bin_base;
		}
		for (int j = k + 1;j < a[i].size();j++) {
			unsigned long long bin_base = a[i][j] * base[j];
			bin_1_tool += bin_base;
			bin_4_tool += bin_base;
		}
		sum.push_back(bin_1_tool);
		sum.push_back(bin_2_tool);
		sum.push_back(bin_3_tool);
		sum.push_back(bin_4_tool);
		bin_1[sum]++;
		bin_2[bin_2_tool]++;
		bin_3[bin_3_tool]++;
		bin_4[bin_4_tool]++;
	}
	double TE = 0;
	for (auto it = bin_1.begin();it != bin_1.end();it++) {
		unsigned long long py_flag = (it->first)[1];
		unsigned long long pxy_flag = (it->first)[2];
		unsigned long long pyx2_flag = (it->first)[3];
		double pyy2x = 1.0 * it->second / data_sum + 1e-8;
		double py2 = 1.0 * bin_2[py_flag] / data_sum + 1e-8;
		double pyy2 = 1.0 * bin_3[pxy_flag] / data_sum + 1e-8;
		double py2x = 1.0 * bin_4[pyx2_flag] / data_sum + 1e-8;

		TE += pyy2x * log2(pyy2x * py2 / (pyy2 * py2x));
	}

	return TE;
}



//广义熵
double trantransfer_Renyi_entropy(vector<double> X, vector<double> Y, int delay, int k, int l, double Alp) {
	if (Alp == 1)
		return -1;

	if (X.size() != Y.size())return 0;
	vector<int> X_prework;
	vector<int> Y_prework;
	X_prework = prework(X);
	Y_prework = prework(Y);

	auto a = get_matru(X_prework, Y_prework, delay, k, l);
	//print_matru(a);

	map<vector<unsigned long long>, int> bin_1;
	map<unsigned long long, int> bin_2;
	map<unsigned long long, int> bin_3;
	map<unsigned long long, int> bin_4;
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
		vector<unsigned long long> sum;
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
		sum.push_back(bin_1_tool);
		sum.push_back(bin_2_tool);
		sum.push_back(bin_3_tool);
		sum.push_back(bin_4_tool);
		bin_1[sum]++;
		bin_2[bin_2_tool]++;
		bin_3[bin_3_tool]++;
		bin_4[bin_4_tool]++;
	}

	//计算熵 广义熵


	double TE = 0;
	//bin_1_p pyyx2  bin_2_p py bin_3_p pxy bin_4_p pyx2
	double bin_1_p = 0, bin_2_p = 0, bin_3_p = 0, bin_4_p = 0;
	for (auto it = bin_1.begin();it != bin_1.end();it++) {
		double p = 1.0 * it->second / data_sum;
		bin_1_p += pow(p, Alp);
	}
	for (auto it = bin_2.begin();it != bin_2.end();it++) {
		double p = 1.0 * it->second / data_sum;
		bin_2_p += pow(p, Alp);
	}
	for (auto it = bin_3.begin();it != bin_3.end();it++) {
		double p = 1.0 * it->second / data_sum;
		bin_3_p += pow(p, Alp);
	}
	for (auto it = bin_4.begin();it != bin_4.end();it++) {
		double p = 1.0 * it->second / data_sum;
		bin_4_p += pow(p, Alp);
	}
	TE = log2(bin_3_p * bin_4_p / (bin_1_p * bin_2_p)) / (1 - Alp);

	return TE;

}

//复数计算转移熵

double trantransfer_entropy_Complex(vector<Complex> X, vector<Complex> Y, int delay, int k, int l) {
	if (X.size() != Y.size())return 0;
	vector<pair<int, int>> X_prework;
	vector<pair<int, int>> Y_prework;
	X_prework = prework(X);
	Y_prework = prework(Y);

	auto a = get_matru(X_prework, Y_prework, delay, k, l);
	//print_matru(a);

	map<vector<unsigned long long>, int> bin_1;
	map<unsigned long long, int> bin_2;
	map<unsigned long long, int> bin_3;
	map<unsigned long long, int> bin_4;
	vector<unsigned long long> base;
	base.push_back(1);
	unsigned long long base_flag = 1;
	const unsigned long long base_num = 1e9 + 7;

	int op_sum = 2 * (1 + k + l);
	for (int i = 1;i <= op_sum;i++) {
		base_flag = base_flag * base_num;
		base.push_back(base_flag);
	}

	int data_sum = a.size();
	for (int i = 0;i < a.size();i++) {
		vector<unsigned long long> sum;
		unsigned long long bin_1_tool = 0, bin_2_tool = 0, bin_3_tool = 0, bin_4_tool = 0;
		unsigned long long bin_base = base[1] * a[i][0].second + a[i][0].first;
		bin_1_tool += bin_base;
		bin_3_tool += bin_base;
		for (int j = 1;j <= k;j++) {
			unsigned long long bin_base = base[2 * j] * a[i][j].first + base[2 * j + 1] * a[i][j].second;
			bin_1_tool += bin_base;
			bin_2_tool += bin_base;
			bin_3_tool += bin_base;
			bin_4_tool += bin_base;
		}
		for (int j = k + 1;j < a[i].size();j++) {
			unsigned long long bin_base = base[2 * j] * a[i][j].first + base[2 * j + 1] * a[i][j].second;
			bin_1_tool += bin_base;
			bin_4_tool += bin_base;
		}
		sum.push_back(bin_1_tool);
		sum.push_back(bin_2_tool);
		sum.push_back(bin_3_tool);
		sum.push_back(bin_4_tool);
		bin_1[sum]++;
		bin_2[bin_2_tool]++;
		bin_3[bin_3_tool]++;
		bin_4[bin_4_tool]++;
	}
	double TE = 0;
	for (auto it = bin_1.begin();it != bin_1.end();it++) {
		unsigned long long py_flag = (it->first)[1];
		unsigned long long pxy_flag = (it->first)[2];
		unsigned long long pyx2_flag = (it->first)[3];
		double pyy2x = 1.0 * it->second / data_sum + 1e-8;
		double py2 = 1.0 * bin_2[py_flag] / data_sum + 1e-8;
		double pyy2 = 1.0 * bin_3[pxy_flag] / data_sum + 1e-8;
		double py2x = 1.0 * bin_4[pyx2_flag] / data_sum + 1e-8;

		TE += pyy2x * log2(pyy2x * py2 / (pyy2 * py2x));
	}

	return TE;
}

//计算广义熵
double trantransfer_Renyi_entropy_Complex(vector<Complex> X, vector<Complex> Y, int delay, int k, int l, double Alp) {
	if (Alp == 1)
		return -1;

	if (X.size() != Y.size())return 0;
	vector<pair<int, int>> X_prework;
	vector<pair<int, int>> Y_prework;
	X_prework = prework(X);
	Y_prework = prework(Y);

	auto a = get_matru(X_prework, Y_prework, delay, k, l);
	//print_matru(a);

	map<vector<unsigned long long>, int> bin_1;
	map<unsigned long long, int> bin_2;
	map<unsigned long long, int> bin_3;
	map<unsigned long long, int> bin_4;
	vector<unsigned long long> base;
	base.push_back(1);
	unsigned long long base_flag = 1;
	const unsigned long long base_num = 1e9 + 7;

	int op_sum = 2 * (1 + k + l);
	for (int i = 1;i <= op_sum;i++) {
		base_flag = base_flag * base_num;
		base.push_back(base_flag);
	}

	int data_sum = a.size();
	for (int i = 0;i < a.size();i++) {
		vector<unsigned long long> sum;
		unsigned long long bin_1_tool = 0, bin_2_tool = 0, bin_3_tool = 0, bin_4_tool = 0;
		unsigned long long bin_base = base[1] * a[i][0].second + a[i][0].first;
		bin_1_tool += bin_base;
		bin_3_tool += bin_base;
		for (int j = 1;j <= k;j++) {
			unsigned long long bin_base = base[2 * j] * a[i][j].first + base[2 * j + 1] * a[i][j].second;
			bin_1_tool += bin_base;
			bin_2_tool += bin_base;
			bin_3_tool += bin_base;
			bin_4_tool += bin_base;
		}
		for (int j = k + 1;j < a[i].size();j++) {
			unsigned long long bin_base = base[2 * j] * a[i][j].first + base[2 * j + 1] * a[i][j].second;
			bin_1_tool += bin_base;
			bin_4_tool += bin_base;
		}
		sum.push_back(bin_1_tool);
		sum.push_back(bin_2_tool);
		sum.push_back(bin_3_tool);
		sum.push_back(bin_4_tool);
		bin_1[sum]++;
		bin_2[bin_2_tool]++;
		bin_3[bin_3_tool]++;
		bin_4[bin_4_tool]++;
	}

	//计算熵 广义熵


	double TE = 0;
	//bin_1_p pyyx2  bin_2_p py bin_3_p pxy bin_4_p pyx2
	double bin_1_p = 0, bin_2_p = 0, bin_3_p = 0, bin_4_p = 0;
	for (auto it = bin_1.begin();it != bin_1.end();it++) {
		double p = 1.0 * it->second / data_sum;
		bin_1_p += pow(p, Alp);
	}
	for (auto it = bin_2.begin();it != bin_2.end();it++) {
		double p = 1.0 * it->second / data_sum;
		bin_2_p += pow(p, Alp);
	}
	for (auto it = bin_3.begin();it != bin_3.end();it++) {
		double p = 1.0 * it->second / data_sum;
		bin_3_p += pow(p, Alp);
	}
	for (auto it = bin_4.begin();it != bin_4.end();it++) {
		double p = 1.0 * it->second / data_sum;
		bin_4_p += pow(p, Alp);
	}
	TE = log2(bin_3_p * bin_4_p / (bin_1_p * bin_2_p)) / (1 - Alp);

	return TE;


}