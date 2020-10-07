#pragma once

#include <vector>
#include <tuple>
#include <cmath>

#include "fftw3.h"

#include "utilities/omp_macros.h"
#include "utilities/c_array.h"
#include "utilities/assume_hint.h"
#include "utilities/alignment.h"

//Note that this requries size_t % (alignment/8)==0
inline void set_values(double* Dst, size_t size, double val) {
	constexpr size_t alignment = Alignment::value;
	__assume(size % (alignment / 8) == 0);
#pragma omp parallel for simd aligned(Dst:alignment) OMP_SIMDLEN_SCHEDULE
	for (size_t itr = 0; itr < size; ++itr) {
		Dst[itr] = val;
	}
}

//Note that this requries range[2] % (alignment/8)==0
//or the behavior will be undefined
//if if_cuthalf is set to be true, only elements with z falls within (0, ((Nz / 2 + 1) / (alignment / 8) + 1) * (alignment / 8)) will be modified
inline void twiddle_factor(const std::vector<std::tuple<double*, double*, c_array<double, 3>>> & data, c_array<size_t, 3> range, bool if_cuthalf = false) {
	constexpr size_t alignment = Alignment::value;
	size_t Nx = range[0];
	size_t Ny = range[1];
	size_t Nz = range[2];
	__assume(Nz % (alignment / 8) == 0);
	const size_t Nz_half = if_cuthalf ? ((Nz / 2 + 1) / (alignment / 8) + 1) * (alignment / 8) : Nz;

#pragma omp parallel
	for (size_t itr_task = 0; itr_task < data.size(); ++itr_task) {
		auto dstcos = std::get<0>(data.at(itr_task));
		auto dstsin = std::get<1>(data.at(itr_task));
		auto paras = std::get<2>(data.at(itr_task));
#pragma omp for OMP_SCHEDULE nowait
		for (size_t itr_x = 0; itr_x < Nx; ++itr_x) {
			for (size_t itr_y = 0; itr_y < Ny; ++itr_y) {
#pragma omp simd aligned(dstcos,dstsin:alignment) OMP_SIMDLEN
				for (size_t itr_z = 0; itr_z < Nz_half; ++itr_z) {
					double angle = itr_x * paras[0] + itr_y * paras[1] + itr_z * paras[2];
					dstcos[(itr_x * Ny + itr_y) * Nz + itr_z] = std::cos(angle);
					dstsin[(itr_x * Ny + itr_y) * Nz + itr_z] = std::sin(angle);
				}
			}
		}
	}

};

//g: translation part for -z,-y,-x respectively
inline std::vector<c_array<double, 3>> generate_m3_operation(c_array<double, 9> g, c_array<size_t, 3> s) {
	return {
		{-2. * Pi * (g[0]),								-2. * Pi * (g[1]),									-2. * Pi * (g[2] - 1. / s[2])					},/*-z,001*/
		{-2. * Pi * (g[3]),								-2. * Pi * (g[4] - 1. / s[1]),						-2. * Pi * (g[5])								},/*-y,010*/
		{-2. * Pi * (g[0] + g[3]),						-2. * Pi * (-g[1] + g[4] - 1. / s[1]),				-2. * Pi * (g[2] + g[5] - 1. / s[2])			},/*-y-z,011*/
		{-2. * Pi * (g[6] - 1. / s[0]),					-2. * Pi * (g[7]),									-2. * Pi * (g[8])								},/*-x,100*/
		{-2. * Pi * (g[6] - g[0] - 1. / s[0]),			-2. * Pi * (g[1] + g[7]),							-2. * Pi * (g[2] + g[8] - 1. / s[2])			},/*-x-z,101*/
		{-2. * Pi * (-g[3] + g[6] - 1. / s[0]),			-2. * Pi * (g[4] + g[7] - 1. / s[1]),				-2. * Pi * (g[5] + g[8])						},/*-x-y,110*/
		{-2. * Pi * (-g[0] - g[3] + g[6] - 1. / s[0]),	-2. * Pi * (-g[1] + g[4] + g[7] - 1. / s[1]),		-2. * Pi * (g[2] + g[5] + g[8] - 1. / s[2])		}/*-x-y-z,111*/
	};
}

//size s is the size of src
//Note that this requries (s[2]/2) % (alignment/8)==0
//src is a full-sized k matrix
//no imaginary part is assumed
//dst has 8 sub matrices, each of which has the size of 1/8 of the original k matrix
//if if_cuthalf is set to be true, only elements with z falls within (0, ((Nz / 4 + 1) / (alignment / 8) + 1) * (alignment / 8)) will be modified
inline void mat_split(double* src, double* const* dst, c_array<size_t, 3> s, bool if_cuthalf = false) {
	constexpr size_t alignment = Alignment::value;
	const size_t Nx = s[0];
	const size_t Ny = s[1];
	const size_t Nz = s[2];
	__assume(Nz % (2 * alignment / 8) == 0);
	const size_t Nz2_half = if_cuthalf ? ((Nz / 4 + 1) / (alignment / 8) + 1) * (alignment / 8) : Nz / 2;

#pragma omp parallel
	{
		for (size_t itr_part = 0; itr_part < 8; ++itr_part) {
			auto local_src = src + ((itr_part & 4) >> 2) * (Nx / 2) * Ny * Nz + ((itr_part & 2) >> 1) * (Ny / 2) * Nz + (itr_part & 1) * (Nz / 2);
			auto local_dst = dst[itr_part];
#pragma omp for OMP_SCHEDULE nowait
			for (size_t itr_x = 0; itr_x < Nx / 2; ++itr_x) {
				for (size_t itr_y = 0; itr_y < Ny / 2; ++itr_y) {
#pragma omp simd aligned(local_src,local_dst:alignment) OMP_SIMDLEN
					for (size_t itr_z = 0; itr_z < Nz2_half; ++itr_z) {
						local_dst[(itr_x * (Ny / 2) + itr_y) * (Nz / 2) + itr_z] = local_src[(itr_x * Ny + itr_y) * Nz + itr_z];
					}
				}
			}
		}
	}
}

//calculate 
//dst[seq[i][0]]=dst[seq[i][0]]+dst[seq[i][1]];
//dst[seq[i][1]]=dst[seq[i][0]]-dst[seq[i][1]];
//size s is the size of each matrix in dst
//if if_cuthalf is set to be true, only elements with z falls within (0, ((Nz / 2 + 1) / (alignment / 8) + 1) * (alignment / 8)) will be modified
inline void add_sub_by_seq(double* const* dst, std::vector<c_array<size_t, 2>> seq, c_array<size_t, 3> s, bool if_cuthalf = false) {
	constexpr size_t alignment = Alignment::value;
	const size_t Nx = s[0];
	const size_t Ny = s[1];
	const size_t Nz = s[2];
	__assume(Nz % (alignment / 8) == 0);
	const size_t NxNy = Nx * Ny;
	const size_t Nz_half = if_cuthalf ? ((Nz / 2 + 1) / (alignment / 8) + 1) * (alignment / 8) : Nz;

#pragma omp parallel
	{
		for (size_t itr_seq = 0; itr_seq < seq.size(); ++itr_seq) {
			auto dst_0 = dst[seq.at(itr_seq)[0]];
			auto dst_1 = dst[seq.at(itr_seq)[1]];
#pragma omp for OMP_SCHEDULE nowait
			for (size_t itr_xy = 0; itr_xy < NxNy; ++itr_xy) {
#pragma omp simd aligned(dst_0,dst_1:alignment) OMP_SIMDLEN
				for (size_t itr_z = 0; itr_z < Nz_half; ++itr_z) {
					auto temp = dst_0[itr_xy * Nz + itr_z] + dst_1[itr_xy * Nz + itr_z];
					dst_1[itr_xy * Nz + itr_z] = dst_0[itr_xy * Nz + itr_z] - dst_1[itr_xy * Nz + itr_z];
					dst_0[itr_xy * Nz + itr_z] = temp;
				}
			}
		}
	}
}

//calculate 
//dst[seq[i][0]]=dst[seq[i][0]]+dst[seq[i][1]];
//dst[seq[i][1]]=dst[seq[i][0]]-dst[seq[i][1]];
//size s is the size of each matrix in dst
//if if_cuthalf is set to be true, only elements with z falls within (0, ((Nz / 2 + 1) / (alignment / 8) + 1) * (alignment / 8)) will be modified
inline void add_sub_by_seq(double* const* dst, const double* const* src, std::vector<c_array<size_t, 2>> seq, c_array<size_t, 3> s, bool if_cuthalf = false) {
	constexpr size_t alignment = Alignment::value;
	const size_t Nx = s[0];
	const size_t Ny = s[1];
	const size_t Nz = s[2];
	__assume(Nz % (alignment / 8) == 0);
	const size_t NxNy = Nx * Ny;
	const size_t Nz_half = if_cuthalf ? ((Nz / 2 + 1) / (alignment / 8) + 1) * (alignment / 8) : Nz;

#pragma omp parallel
	{
		for (size_t itr_seq = 0; itr_seq < seq.size(); ++itr_seq) {
			auto src_0 = src[seq.at(itr_seq)[0]];
			auto src_1 = src[seq.at(itr_seq)[1]];
			auto dst_0 = dst[seq.at(itr_seq)[0]];
			auto dst_1 = dst[seq.at(itr_seq)[1]];
#pragma omp for OMP_SCHEDULE nowait
			for (size_t itr_xy = 0; itr_xy < NxNy; ++itr_xy) {
#pragma omp simd aligned(dst_0,dst_1:alignment) OMP_SIMDLEN
				for (size_t itr_z = 0; itr_z < Nz_half; ++itr_z) {
					dst_0[itr_xy * Nz + itr_z] = src_0[itr_xy * Nz + itr_z] + src_1[itr_xy * Nz + itr_z];
					dst_1[itr_xy * Nz + itr_z] = src_0[itr_xy * Nz + itr_z] - src_1[itr_xy * Nz + itr_z];
				}
			}
		}
	}
}

inline void mul_add_sub_by_seq(double* const* dst_re, double* const* dst_im, const double* const* src_re, const double* const* src_im, const double* const* src_mul, std::vector<c_array<size_t, 2>> seq, c_array<size_t, 3> s, bool if_cuthalf = false) {
	constexpr size_t alignment = Alignment::value;
	const size_t Nx = s[0];
	const size_t Ny = s[1];
	const size_t Nz = s[2];
	__assume(Nz % (alignment / 8) == 0);
	const size_t NxNy = Nx * Ny;
	const size_t Nz_half = if_cuthalf ? ((Nz / 2 + 1) / (alignment / 8) + 1) * (alignment / 8) : Nz;

#pragma omp parallel
	{
		for (size_t itr_seq = 0; itr_seq < seq.size(); ++itr_seq) {
			auto src_re_0 = src_re[seq.at(itr_seq)[0]];
			auto src_re_1 = src_re[seq.at(itr_seq)[1]];
			auto src_im_0 = src_im[seq.at(itr_seq)[0]];
			auto src_im_1 = src_im[seq.at(itr_seq)[1]];
			auto dst_re_0 = dst_re[seq.at(itr_seq)[0]];
			auto dst_re_1 = dst_re[seq.at(itr_seq)[1]];
			auto dst_im_0 = dst_im[seq.at(itr_seq)[0]];
			auto dst_im_1 = dst_im[seq.at(itr_seq)[1]];
			auto mul_0 = src_mul[seq.at(itr_seq)[0]];
			auto mul_1 = src_mul[seq.at(itr_seq)[1]];
#pragma omp for OMP_SCHEDULE nowait
			for (size_t itr_xy = 0; itr_xy < NxNy; ++itr_xy) {
#pragma omp simd aligned(src_re_0,src_re_1,src_im_0,src_im_1,dst_re_0,dst_re_1,dst_im_0,dst_im_1,mul_0,mul_1:alignment) OMP_SIMDLEN
				for (size_t itr_z = 0; itr_z < Nz_half; ++itr_z) {
					dst_re_0[itr_xy * Nz + itr_z] = mul_0[itr_xy * Nz + itr_z] * src_re_0[itr_xy * Nz + itr_z];
					dst_im_0[itr_xy * Nz + itr_z] = mul_0[itr_xy * Nz + itr_z] * src_im_0[itr_xy * Nz + itr_z];
					dst_re_1[itr_xy * Nz + itr_z] = mul_1[itr_xy * Nz + itr_z] * src_re_1[itr_xy * Nz + itr_z];
					dst_im_1[itr_xy * Nz + itr_z] = mul_1[itr_xy * Nz + itr_z] * src_im_1[itr_xy * Nz + itr_z];

					auto tempre = dst_re_0[itr_xy * Nz + itr_z] + dst_re_1[itr_xy * Nz + itr_z];
					dst_re_1[itr_xy * Nz + itr_z] = dst_re_0[itr_xy * Nz + itr_z] - dst_re_1[itr_xy * Nz + itr_z];
					dst_re_0[itr_xy * Nz + itr_z] = tempre;

					auto tempim = dst_im_0[itr_xy * Nz + itr_z] + dst_im_1[itr_xy * Nz + itr_z];
					dst_im_1[itr_xy * Nz + itr_z] = dst_im_0[itr_xy * Nz + itr_z] - dst_im_1[itr_xy * Nz + itr_z];
					dst_im_0[itr_xy * Nz + itr_z] = tempim;

				}
			}
		}
	}
}

//conduct the radix-2 Cooley-Turkey algorithm for real FFT
//Hermtian symmetry is make used of to ensure all access at z-direction,
//namely the innermost loop is always contiguous
//size: the size of the sub Fourier transform, (x,y,z)
//number: the range required to calculated, (x,y,z), number<size
//src_stride: the embedded size of the src matrix, (y,z), since only one radix-2 Cooley-Turkey algorithm is allowed
//dst_stride: the embedded size of the dst matrix, (y,z), since only one radix-2 Cooley-Turkey algorithm is allowed
//twiddle_stride: the embedded size of the twiddle factor, (y,z), since only one radix-2 Cooley-Turkey algorithm is allowed
//size<src_stride, size<dst_stride, size<twiddle_stride
template<size_t Index>
void real_radix_2_element_3d(
	const fftw_complex* src, fftw_complex* dst, const double* const* twiddle_re, const double* const* twiddle_im,
	c_array<size_t, 3> size, c_array<size_t, 3> number, c_array<size_t, 2> src_stride, c_array<size_t, 2> dst_stride, c_array<size_t, 2> twiddle_stride
) {
	constexpr size_t IndexPos0 = (Index & 0x1) >> 0;
	constexpr size_t IndexPos1 = (Index & 0x2) >> 1;
	constexpr size_t IndexPos2 = (Index & 0x4) >> 2;
	constexpr bool IfSwitch = (IndexPos0 + IndexPos1 + IndexPos2) % 2 == 1;
	constexpr size_t TwiddleLeftIndex0 = IfSwitch ? 7 : 0;
	constexpr size_t TwiddleLeftIndex1 = IfSwitch ? 1 : 6;
	constexpr size_t TwiddleLeftIndex2 = IfSwitch ? 5 : 2;
	constexpr size_t TwiddleLeftIndex3 = IfSwitch ? 3 : 4;
	constexpr size_t TwiddleRightIndex0 = IfSwitch ? 0 : 7;
	constexpr size_t TwiddleRightIndex1 = IfSwitch ? 6 : 1;
	constexpr size_t TwiddleRightIndex2 = IfSwitch ? 2 : 5;
	constexpr size_t TwiddleRightIndex3 = IfSwitch ? 4 : 3;
	constexpr bool IfNegativeIndex0 = ((IndexPos2 & 0) + (IndexPos1 & 0) + (IndexPos0 & 0)) % 2 == 1;
	constexpr bool IfNegativeIndex1 = ((IndexPos2 & 1) + (IndexPos1 & 1) + (IndexPos0 & 0)) % 2 == 1;
	constexpr bool IfNegativeIndex2 = ((IndexPos2 & 0) + (IndexPos1 & 1) + (IndexPos0 & 0)) % 2 == 1;
	constexpr bool IfNegativeIndex3 = ((IndexPos2 & 1) + (IndexPos1 & 0) + (IndexPos0 & 0)) % 2 == 1;
	constexpr double FactorIndex0 = IfNegativeIndex0 ? -1 : 1;
	constexpr double FactorIndex1 = IfNegativeIndex1 ? -1 : 1;
	constexpr double FactorIndex2 = IfNegativeIndex2 ? -1 : 1;
	constexpr double FactorIndex3 = IfNegativeIndex3 ? -1 : 1;

	for (size_t itr_x = 0; itr_x < number[0]; ++itr_x) {
		for (size_t itr_y = 0; itr_y < number[1]; ++itr_y) {
			auto twiddle_shift = (itr_x * twiddle_stride[0] + itr_y) * twiddle_stride[1];
			auto src000 = src + (itr_x * src_stride[0] + itr_y) * src_stride[1];
			auto src110 = src + (((size[0] - itr_x) % size[0]) * src_stride[0] + (size[1] - itr_y) % size[1]) * src_stride[1];
			auto src010 = src + (itr_x * src_stride[0] + (size[1] - itr_y) % size[1]) * src_stride[1];
			auto src100 = src + (((size[0] - itr_x) % size[0]) * src_stride[0] + itr_y) * src_stride[1];
			auto dstxxx = dst + (itr_x * dst_stride[0] + itr_y) * dst_stride[1];
			for (size_t itr_z = 0; itr_z < number[2]; ++itr_z) {
				dstxxx[itr_z][0]
					= FactorIndex0 * (src000[itr_z][0] * twiddle_re[TwiddleLeftIndex0][twiddle_shift + itr_z] - src000[itr_z][1] * twiddle_im[TwiddleRightIndex0][twiddle_shift + itr_z])
					+ FactorIndex1 * (src110[itr_z][0] * twiddle_re[TwiddleLeftIndex1][twiddle_shift + itr_z] - src110[itr_z][1] * twiddle_im[TwiddleRightIndex1][twiddle_shift + itr_z])
					+ FactorIndex2 * (src010[itr_z][0] * twiddle_re[TwiddleLeftIndex2][twiddle_shift + itr_z] - src010[itr_z][1] * twiddle_im[TwiddleRightIndex2][twiddle_shift + itr_z])
					+ FactorIndex3 * (src100[itr_z][0] * twiddle_re[TwiddleLeftIndex3][twiddle_shift + itr_z] - src100[itr_z][1] * twiddle_im[TwiddleRightIndex3][twiddle_shift + itr_z]);
				dstxxx[itr_z][1]
					= FactorIndex0 * (src000[itr_z][0] * twiddle_im[TwiddleLeftIndex0][twiddle_shift + itr_z] + src000[itr_z][1] * twiddle_re[TwiddleRightIndex0][twiddle_shift + itr_z])
					+ FactorIndex1 * (src110[itr_z][0] * twiddle_im[TwiddleLeftIndex1][twiddle_shift + itr_z] + src110[itr_z][1] * twiddle_re[TwiddleRightIndex1][twiddle_shift + itr_z])
					+ FactorIndex2 * (src010[itr_z][0] * twiddle_im[TwiddleLeftIndex2][twiddle_shift + itr_z] + src010[itr_z][1] * twiddle_re[TwiddleRightIndex2][twiddle_shift + itr_z])
					+ FactorIndex3 * (src100[itr_z][0] * twiddle_im[TwiddleLeftIndex3][twiddle_shift + itr_z] + src100[itr_z][1] * twiddle_re[TwiddleRightIndex3][twiddle_shift + itr_z]);
			}
		}
	}
}

template<typename T>
inline void make_matrix(T* __restrict* memptr, size_t size, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr)
{
	(*memptr) = (T*)_mm_malloc(size * sizeof(T), Alignment::value);
}
template<typename T>
inline void destroy_matrix(T* __restrict* memptr, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr)
{
	_mm_free(*memptr);
	*memptr = nullptr;
}
template<typename T>
inline void make_matrix_list(T* __restrict* memptr, size_t size, size_t number, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr)
{
	make_matrix<T>(memptr, size * number);
	for (size_t itr_number = 0; itr_number < number; ++itr_number)
		memptr[itr_number] = memptr[0] + itr_number * size;
}
template<typename T>
inline void destory_matrix_list(T* __restrict* memptr, size_t number, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr)
{
	destroy_matrix<T>(memptr);
	for (size_t itr_number = 0; itr_number < number; ++itr_number)
		memptr[itr_number] = nullptr;
}
