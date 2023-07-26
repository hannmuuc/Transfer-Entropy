#include"pch.h"
#include"FFT.h"


std::ostream& operator << (std::ostream& out, const Complex_& s)
{
	out << s.real << "+" << s.imagin << "j";
	return out;
}


int isBase2(int size_n)
{
	int k = size_n;
	int z = 0;
	while (k /= 2)
	{
		z++;
	}
	k = z;
	if (size_n != (1 << k))
		return -1;
	else
		return k;
}

void Add_Complex(Complex* src1, Complex* src2, Complex* dst)
{
	dst->imagin = src1->imagin + src2->imagin;
	dst->real = src1->real + src2->real;
}
void Sub_Complex(Complex* src1, Complex* src2, Complex* dst)
{
	dst->imagin = src1->imagin - src2->imagin;
	dst->real = src1->real - src2->real;
}
void Multy_Complex(Complex* src1, Complex* src2, Complex* dst)
{
	double r1 = 0.0, r2 = 0.0;
	double i1 = 0.0, i2 = 0.0;
	r1 = src1->real;
	r2 = src2->real;
	i1 = src1->imagin;
	i2 = src2->imagin;
	dst->imagin = r1 * i2 + r2 * i1;
	dst->real = r1 * r2 - i1 * i2;
}
void Copy_Complex(Complex* src, Complex* dst)
{
	dst->real = src->real;
	dst->imagin = src->imagin;
}

void getWN(double n, double size_n, Complex* dst)
{
	double x = 2.0 * 3.14159265358979323846 * n / size_n;
	dst->imagin = -sin(x);
	dst->real = cos(x);
}

int FFTReal_remap(double* src, int size_n)
{

	if (size_n == 1)
		return 0;
	double* temp = (double*)malloc(sizeof(double) * size_n);
	for (int i = 0; i < size_n; i++)
		if (i % 2 == 0)
			temp[i / 2] = src[i];
		else
			temp[(size_n + i) / 2] = src[i];
	for (int i = 0; i < size_n; i++)
		src[i] = temp[i];
	free(temp);
	FFTReal_remap(src, size_n / 2);
	FFTReal_remap(src + size_n / 2, size_n / 2);
	return 1;
}
int FFTComplex_remap(Complex* src, int size_n)
{

	if (size_n == 1)
		return 0;
	Complex* temp = (Complex*)malloc(sizeof(Complex) * size_n);
	for (int i = 0; i < size_n; i++)
		if (i % 2 == 0)
			Copy_Complex(&src[i], &(temp[i / 2]));
		else
			Copy_Complex(&(src[i]), &(temp[(size_n + i) / 2]));
	for (int i = 0; i < size_n; i++)
		Copy_Complex(&(temp[i]), &(src[i]));
	free(temp);
	FFTComplex_remap(src, size_n / 2);
	FFTComplex_remap(src + size_n / 2, size_n / 2);
	return 1;
}

void FFT(Complex* src, Complex* dst, int size_n)
{

	int k = size_n;
	int z = 0;
	while (k /= 2)
	{
		z++;
	}
	k = z;
	if (size_n != (1 << k))
		exit(0);
	Complex* src_com = (Complex*)malloc(sizeof(Complex) * size_n);
	if (src_com == NULL)
		exit(0);
	for (int i = 0; i < size_n; i++)
	{
		Copy_Complex(&src[i], &src_com[i]);
	}
	FFTComplex_remap(src_com, size_n);
	for (int i = 0; i < k; i++)
	{
		z = 0;
		for (int j = 0; j < size_n; j++)
		{
			if ((j / (1 << i)) % 2 == 1)
			{
				Complex wn;
				getWN(z, size_n, &wn);
				Multy_Complex(&src_com[j], &wn, &src_com[j]);
				z += 1 << (k - i - 1);
				Complex temp;
				int neighbour = j - (1 << (i));
				temp.real = src_com[neighbour].real;
				temp.imagin = src_com[neighbour].imagin;
				Add_Complex(&temp, &src_com[j], &src_com[neighbour]);
				Sub_Complex(&temp, &src_com[j], &src_com[j]);
			}
			else
				z = 0;
		}
	}

	for (int i = 0; i < size_n; i++)
	{
		Copy_Complex(&src_com[i], &dst[i]);
	}
	free(src_com);
}
void RealFFT(double* src, Complex* dst, int size_n)
{

	int k = size_n;
	int z = 0;
	while (k /= 2)
	{
		z++;
	}
	k = z;
	if (size_n != (1 << k))
		exit(0);
	Complex* src_com = (Complex*)malloc(sizeof(Complex) * size_n);
	if (src_com == NULL)
		exit(0);
	for (int i = 0; i < size_n; i++)
	{
		src_com[i].real = src[i];
		src_com[i].imagin = 0;
	}
	FFTComplex_remap(src_com, size_n);
	for (int i = 0; i < k; i++)
	{
		z = 0;
		for (int j = 0; j < size_n; j++)
		{
			if ((j / (1 << i)) % 2 == 1)
			{
				Complex wn;
				getWN(z, size_n, &wn);
				Multy_Complex(&src_com[j], &wn, &src_com[j]);
				z += 1 << (k - i - 1);
				Complex temp;
				int neighbour = j - (1 << (i));
				temp.real = src_com[neighbour].real;
				temp.imagin = src_com[neighbour].imagin;
				Add_Complex(&temp, &src_com[j], &src_com[neighbour]);
				Sub_Complex(&temp, &src_com[j], &src_com[j]);
			}
			else
				z = 0;
		}
	}

	for (int i = 0; i < size_n; i++)
	{
		Copy_Complex(&src_com[i], &dst[i]);
	}
	free(src_com);
}

void IFFT(Complex* src, Complex* dst, int size_n)
{
	for (int i = 0; i < size_n; i++)
		src[i].imagin = -src[i].imagin;
	FFTComplex_remap(src, size_n);
	int z, k;
	if ((z = isBase2(size_n)) != -1)
		k = isBase2(size_n);
	else
		exit(0);
	for (int i = 0; i < k; i++)
	{
		z = 0;
		for (int j = 0; j < size_n; j++)
		{
			if ((j / (1 << i)) % 2 == 1)
			{
				Complex wn;
				getWN(z, size_n, &wn);
				Multy_Complex(&src[j], &wn, &src[j]);
				z += 1 << (k - i - 1);
				Complex temp;
				int neighbour = j - (1 << (i));
				Copy_Complex(&src[neighbour], &temp);
				Add_Complex(&temp, &src[j], &src[neighbour]);
				Sub_Complex(&temp, &src[j], &src[j]);
			}
			else
				z = 0;
		}
	}
	for (int i = 0; i < size_n; i++)
	{
		dst[i].imagin = (1. / size_n) * src[i].imagin;
		dst[i].real = (1. / size_n) * src[i].real;
	}
}

int DFT2D(double* src, Complex* dst, int size_w, int size_h) {
	for (int u = 0;u < size_w;u++) {
		for (int v = 0;v < size_h;v++) {
			double real = 0.0;
			double imagin = 0.0;
			for (int i = 0;i < size_w;i++) {
				for (int j = 0;j < size_h;j++) {
					double I = src[i * size_w + j];
					double x = 3.14159265358979323846 * 2 * ((double)i * u / (double)size_w + (double)j * v / (double)size_h);
					real += cos(x) * I;
					imagin += -sin(x) * I;

				}
			}
			dst[u * size_w + v].real = real;
			dst[u * size_w + v].imagin = imagin;

		}

	}
	return 0;
}

int IDFT2D(Complex* src, Complex* dst, int size_w, int size_h) {
	for (int i = 0;i < size_w;i++) {
		for (int j = 0;j < size_h;j++) {
			double real = 0.0;
			double imagin = 0.0;
			for (int u = 0;u < size_w;u++) {
				for (int v = 0;v < size_h;v++) {
					double R = src[u * size_w + v].real;
					double I = src[u * size_w + v].imagin;
					double x = 3.14159265358979323846 * 2 * ((double)i * u / (double)size_w + (double)j * v / (double)size_h);
					real += R * cos(x) - I * sin(x);
					imagin += I * cos(x) + R * sin(x);

				}
			}
			dst[i * size_w + j].real = (1. / (size_w * size_h)) * real;
			dst[i * size_w + j].imagin = (1. / (size_w * size_h)) * imagin;

		}
	}
	return 0;
}

void ColumnVector(Complex* src, Complex* dst, int size_w, int size_h)
{
	for (int i = 0; i < size_h; i++)
		Copy_Complex(&src[size_w * i], &dst[i]);
}

void IColumnVector(Complex* src, Complex* dst, int size_w, int size_h)
{
	for (int i = 0; i < size_h; i++)
		Copy_Complex(&src[i], &dst[size_w * i]);
}

int FFT2D(double* src, Complex* dst, int size_w, int size_h)
{
	if (isBase2(size_w) == -1 || isBase2(size_h) == -1)
		exit(0);
	Complex* temp = (Complex*)malloc(sizeof(Complex) * size_h * size_w);
	if (temp == NULL)
		return -1;
	for (int i = 0; i < size_h; i++)
	{
		RealFFT(&src[size_w * i], &temp[size_w * i], size_w);
	}

	Complex* Column = (Complex*)malloc(sizeof(Complex) * size_h);
	if (Column == NULL)
		return -1;
	for (int i = 0; i < size_w; i++)
	{
		ColumnVector(&temp[i], Column, size_w, size_h);
		FFT(Column, Column, size_h);
		IColumnVector(Column, &temp[i], size_w, size_h);
	}

	for (int i = 0; i < size_h * size_w; i++)
		Copy_Complex(&temp[i], &dst[i]);
	free(temp);
	free(Column);
	return 0;
}

int IFFT2D(Complex* src, Complex* dst, int size_w, int size_h)
{

	if (isBase2(size_w) == -1 || isBase2(size_h) == -1)
		exit(0);

	Complex* temp = (Complex*)malloc(sizeof(Complex) * size_h * size_w);
	if (temp == NULL)
		return -1;
	Complex* Column = (Complex*)malloc(sizeof(Complex) * size_h);
	if (Column == NULL)
		return -1;

	for (int i = 0; i < size_w; i++)
	{
		ColumnVector(&src[i], Column, size_w, size_h);
		IFFT(Column, Column, size_h);
		IColumnVector(Column, &src[i], size_w, size_h);
	}
	for (int i = 0; i < size_w * size_h; i++)
		src[i].imagin = -src[i].imagin;
	for (int i = 0; i < size_h; i++)
	{
		IFFT(&src[size_w * i], &temp[size_w * i], size_w);
	}

	for (int i = 0; i < size_h * size_w; i++)
		Copy_Complex(&temp[i], &dst[i]);
	free(temp);
	free(Column);
	return 0;
}

std::vector < std::vector<Complex>> FFT_2D(std::vector<std::vector<double>>& data) {

	std::vector < std::vector<Complex>> res;

	if (data.size() == 0 || data[0].size() == 0)
		return res;

	int n, m;
	for (int i = 1;i <= data.size();i *= 2) {
		n = i;
	}
	for (int i = 1;i <= data[0].size();i *= 2) {
		m = i;
	}

	double* input = new double[n * m];
	Complex* dst = new Complex[n * m];

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			input[i * m + j] = data[i][j];
		}
	}

	FFT2D(input, dst, m, n);

	for (int i = 0; i < n; i++)
	{
		res.push_back({});
		for (int j = 0; j < m; j++)
		{
			res[i].push_back(dst[i * m + j]);
		}
	}

	return res;
}
