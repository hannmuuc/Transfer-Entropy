#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

//实现复数操作
class Complex_
{
public:
	double real;
	double imagin;
};
typedef struct Complex_ Complex;

//输出
std::ostream& operator << (std::ostream& out, const Complex_& s);
//判断是否以2为底
int isBase2(int size_n);
//复数运算
void Add_Complex(Complex* src1, Complex* src2, Complex* dst);
void Sub_Complex(Complex* src1, Complex* src2, Complex* dst);
void Multy_Complex(Complex* src1, Complex* src2, Complex* dst);
void Copy_Complex(Complex* src, Complex* dst);
void Show_Complex(Complex* src, int size_n);
//处理数据
std::vector <std::vector<Complex>> FFT_2D(std::vector<std::vector<double>>& data);
void getWN(double n, double size_n, Complex* dst);
int FFTReal_remap(double* src, int size_n);
int FFTComplex_remap(Complex* src, int size_n);
//生成2FFT
void FFT(Complex* src, Complex* dst, int size_n);
void RealFFT(double* src, Complex* dst, int size_n);
void IFFT(Complex* src, Complex* dst, int size_n);
void ColumnVector(Complex* src, Complex* dst, int size_w, int size_h);
void IColumnVector(Complex* src, Complex* dst, int size_w, int size_h);
//生出2D-FFT
int DFT2D(double* src, Complex* dst, int size_w, int size_h);
int IDFT2D(Complex* src, Complex* dst, int size_w, int size_h);
int FFT2D(double* src, Complex* dst, int size_w, int size_h);
int IFFT2D(Complex* src, Complex* dst, int size_w, int size_h);
