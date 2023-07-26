#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

//ʵ�ָ�������
class Complex_
{
public:
	double real;
	double imagin;
};
typedef struct Complex_ Complex;

//���
std::ostream& operator << (std::ostream& out, const Complex_& s);
//�ж��Ƿ���2Ϊ��
int isBase2(int size_n);
//��������
void Add_Complex(Complex* src1, Complex* src2, Complex* dst);
void Sub_Complex(Complex* src1, Complex* src2, Complex* dst);
void Multy_Complex(Complex* src1, Complex* src2, Complex* dst);
void Copy_Complex(Complex* src, Complex* dst);
void Show_Complex(Complex* src, int size_n);
//��������
std::vector <std::vector<Complex>> FFT_2D(std::vector<std::vector<double>>& data);
void getWN(double n, double size_n, Complex* dst);
int FFTReal_remap(double* src, int size_n);
int FFTComplex_remap(Complex* src, int size_n);
//����2FFT
void FFT(Complex* src, Complex* dst, int size_n);
void RealFFT(double* src, Complex* dst, int size_n);
void IFFT(Complex* src, Complex* dst, int size_n);
void ColumnVector(Complex* src, Complex* dst, int size_w, int size_h);
void IColumnVector(Complex* src, Complex* dst, int size_w, int size_h);
//����2D-FFT
int DFT2D(double* src, Complex* dst, int size_w, int size_h);
int IDFT2D(Complex* src, Complex* dst, int size_w, int size_h);
int FFT2D(double* src, Complex* dst, int size_w, int size_h);
int IFFT2D(Complex* src, Complex* dst, int size_w, int size_h);
