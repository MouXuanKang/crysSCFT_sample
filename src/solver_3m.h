#pragma once

#include <map>
#include <iostream>

#include "solver_base.h"
#include "solver_utilities.h"

#include "utilities/omp_macros.h"
#include "utilities/c_array.h"
#include "utilities/assume_hint.h"
#include "utilities/alignment.h"

class CrysPSSolver3m : public CrysPSSolverBase {

public:
	CrysPSSolver3m(const c_array<size_t, 3>& grid_size_log, const c_array<double, 9> translational_part = { 0,0,0,0,0,0,0,0,0 })
		: CrysPSSolverBase(grid_size_log), translational_part_(translational_part) {
		set_grid();
	}
	inline ~CrysPSSolver3m() override;
	inline void set_cell_para(const c_array<double, 6>& cell_para) override;
	inline void set_contour_step(double ds) override;
	inline void diffusion() const override;

private:
	double** r_re, ** r_im;//generated once, twiddle factors
	double** k_re, ** k_im;//genereted each time cell_para_ or ds changed, for diffusion

	inline void set_grid();
	std::map<double, std::vector<double**> > K_mat_collection;

	inline void K_mat_create(double ds);
	inline void K_mat_destory();

	size_t Nx, Ny, Nz;
	size_t Nx2_padded, Ny2_padded, Nz2_padded;
	//fftw plans for diffusion
	fftw_plan fft_xyz_3d_r2c_forward;
	fftw_plan fft_xyz_3d_c2r_backward;

	//for diffusion 
	fftw_complex* step1_xyz_out;
	fftw_complex* step2_xyz_in_reverse;

private:
	void R_mat_generator_full() const;
	void K_mat_generator(double** k_re_, double** k_im_, double dz, double Lx, double Ly, double Lz) const;
	void fftw_plan_initial();
	void xyz_routine() const;

	const c_array<double, 9> translational_part_;
};

inline CrysPSSolver3m::~CrysPSSolver3m()
{
	//iomat
	destroy_matrix(&iomat_for_diffusion_);

	//twiddle factors
	destory_matrix_list(r_re, 8);
	delete[] r_re;
	destory_matrix_list(r_im, 8);
	delete[] r_im;

	//for diffusion 
	destroy_matrix((double**)&step1_xyz_out);
	destroy_matrix((double**)&step2_xyz_in_reverse);

	K_mat_destory();
}

inline void CrysPSSolver3m::set_grid()
{
	Nx = grid_size_log_[0];
	Ny = grid_size_log_[1];
	Nz = grid_size_log_[2];
	Nx2_padded = Nx / 2 + 1;
	Ny2_padded = Ny / 2 + 1;
	Nz2_padded = Nz / 2 + Alignment::value / 8;
	if (Nx % 2 || Ny % 2 || Nz % 2) {
		std::cout << "ERROR: Grid Size Must Be Even For Generalized 3m symmetry" << std::endl;
		exit(1);
	}
	if (Nx <= 0 || Ny <= 0 || Nz <= 0) {
		std::cout << "ERROR: Bad Grid Size For Generalized 3m symmetry" << std::endl;
		exit(1);
	}
	if ((Nz / 2) % (Alignment::value / 8) != 0) {
		std::cout << "ERROR: Bad Grid Size For Generalized 3m symmetry, alignment cannot be guaranteed" << std::endl;
		exit(1);
	}

	grid_size_phy_ = { Nx / 2 ,Ny / 2 ,Nz / 2 };
	space_size_phy_ = Nx * Ny * Nz / 8;

	//iomat
	make_matrix(&iomat_for_diffusion_, space_size_phy_);

	//twiddle factors
	r_re = new double* [8];
	make_matrix_list(r_re, space_size_phy_, 8);
	r_im = new double* [8];
	make_matrix_list(r_im, space_size_phy_, 8);

	//for diffusion 
	make_matrix((double**)&step1_xyz_out, 2 * space_size_phy_);
	make_matrix((double**)&step2_xyz_in_reverse, 2 * space_size_phy_);

	R_mat_generator_full();

	fftw_plan_initial();
}

inline void CrysPSSolver3m::set_cell_para(const c_array<double, 6>& cell_para)
{
	if (cell_para[0] == cell_para_[0] && cell_para[1] == cell_para_[1] && cell_para[2] == cell_para_[2] && cell_para[3] == 0 && cell_para[4] == 0 && cell_para[5] == 0) return;
	cell_para_ = { cell_para[0],cell_para[1],cell_para[2],0,0,0 };
	K_mat_destory();
	ds_current_ = 0.00;
}

inline void CrysPSSolver3m::set_contour_step(double ds)
{
	if (K_mat_collection.size() > 0 && ds == ds_current_) {
		return;
	}

	ds_current_ = ds;

	auto target = K_mat_collection.find(ds);

	if (target != K_mat_collection.end()) {
		k_re = target->second.at(0);
		k_im = target->second.at(1);
	}
	else {
		K_mat_create(ds);
		K_mat_generator(k_re, k_im, ds, cell_para_[0], cell_para_[1], cell_para_[2]);
	}
}

inline void CrysPSSolver3m::diffusion() const
{
	xyz_routine();
}

inline void CrysPSSolver3m::K_mat_create(double ds)
{
	//for diffusion
	k_re = new double* [8];
	make_matrix_list(k_re, space_size_phy_, 8);
	k_im = new double* [8];
	make_matrix_list(k_im, space_size_phy_, 8);

	K_mat_collection.insert(std::make_pair(ds, std::vector<double**>{k_re, k_im}));
}

inline void CrysPSSolver3m::K_mat_destory()
{
	for (const auto& K_ele : K_mat_collection) {
		//for diffusion
		destory_matrix_list(K_ele.second.at(0), 8);
		delete[] K_ele.second.at(0);
		destory_matrix_list(K_ele.second.at(1), 8);
		delete[] K_ele.second.at(1);
	}

	K_mat_collection.clear();
}

inline void CrysPSSolver3m::R_mat_generator_full() const
{
	const c_array<size_t, 3> size = { Nx,Ny,Nz };
	const c_array<size_t, 3> halfsize = { Nx / 2,Ny / 2,Nz / 2 };
	auto operations = generate_m3_operation(translational_part_, size);
	std::vector<std::tuple<double*, double*, c_array<double, 3>>> twiddle;
	for (size_t itr = 0; itr < operations.size(); ++itr) twiddle.push_back(std::make_tuple(r_re[itr + 1], r_im[itr + 1], operations.at(itr)));
	set_values(r_re[0], space_size_phy_, 1);
	set_values(r_im[0], space_size_phy_, 0);
	twiddle_factor(twiddle, halfsize, false);
}

inline void CrysPSSolver3m::K_mat_generator(double** k_re_, double** k_im_, double dz, double Lx, double Ly, double Lz) const
{
	constexpr size_t alignment = Alignment::value;
	__assume(Nz % (2 * (alignment / 8)) == 0);

	const c_array<size_t, 3> size = { Nx,Ny,Nz };
	const c_array<size_t, 3> halfsize = { Nx / 2,Ny / 2,Nz / 2 };

	double* __restrict kx, * __restrict ky, * __restrict kz;
	make_matrix(&kx, Nx);
	make_matrix(&ky, Ny);
	make_matrix(&kz, Nz);

	double factor_Lx = 2. * Pi / Lx;
	factor_Lx *= factor_Lx;
	double factor_Ly = 2. * Pi / Ly;
	factor_Ly *= factor_Ly;
	double factor_Lz = 2. * Pi / Lz;
	factor_Lz *= factor_Lz;

#pragma omp simd aligned(kx:alignment) OMP_SIMDLEN
	for (size_t itr_x = 0; itr_x < Nx; ++itr_x) {
		auto temp = itr_x > (Nx / 2) ? (Nx - itr_x) : itr_x;
		kx[itr_x] = temp * temp * factor_Lx;
	}
#pragma omp simd aligned(ky:alignment) OMP_SIMDLEN
	for (size_t itr_y = 0; itr_y < Ny; ++itr_y)
	{
		auto temp = itr_y > (Ny / 2) ? (Ny - itr_y) : itr_y;
		ky[itr_y] = temp * temp * factor_Ly;
	}
#pragma omp simd aligned(kz:alignment) OMP_SIMDLEN
	for (size_t itr_z = 0; itr_z < Nz; ++itr_z) {
		auto temp = itr_z > (Nz / 2) ? (Nz - itr_z) : itr_z;
		kz[itr_z] = temp * temp * factor_Lz;
	}
	double* __restrict tempmat;
	make_matrix(&tempmat, space_size_log_);

	double factor = 1. / space_size_log_;
#pragma omp parallel
	{
#pragma omp	for OMP_SCHEDULE
		for (size_t itr_x = 0; itr_x < Nx; ++itr_x) {
			for (size_t itr_y = 0; itr_y < Ny; ++itr_y) {
#pragma omp simd aligned(kx,ky,kz,tempmat:alignment) OMP_SIMDLEN
				for (size_t itr_z = 0; itr_z < Nz; ++itr_z) {
					tempmat[(itr_x * Ny + itr_y) * Nz + itr_z] = exp(-(kx[itr_x] + ky[itr_y] + kz[itr_z]) * dz) * factor;
				}
			}
		}
	}
	destroy_matrix(&kx);
	destroy_matrix(&ky);
	destroy_matrix(&kz);

	double** k_split = new double* [8];
	make_matrix_list(k_split, space_size_phy_, 8);
	mat_split(tempmat, k_split, { Nx,Ny,Nz }, true);

	add_sub_by_seq(k_split, { {0,4},{1,5},{2,6},{3,7} }, halfsize, true);
	add_sub_by_seq(k_split, { {0,2},{1,3},{4,6},{5,7} }, halfsize, true);
	add_sub_by_seq(k_split, { {0,1},{2,3},{4,5},{6,7} }, halfsize, true);

	mul_add_sub_by_seq(k_re, k_im, r_re, r_im, k_split, {
		{0,7},{6,1},{2,5},{4,3}
		}, halfsize, true);

	destory_matrix_list(k_split, 8);
	delete[] k_split;
	destroy_matrix(&tempmat);
}

inline void CrysPSSolver3m::fftw_plan_initial()
{
	int rank1 = 3;
	fftw_iodim dims1[3];
	int howmany_rank1 = 0;
	fftw_iodim howmany_dims1[1];

	dims1[0].n = (int)Nx / 2;
	dims1[0].is = ((int)Ny / 2) * ((int)Nz / 2);
	dims1[0].os = ((int)Ny / 2) * ((int)Nz / 2);
	dims1[1].n = (int)Ny / 2;
	dims1[1].is = ((int)Nz / 2);
	dims1[1].os = ((int)Nz / 2);
	dims1[2].n = (int)Nz / 2;
	dims1[2].is = 1;
	dims1[2].os = 1;
	howmany_dims1[0].n = 1;
	howmany_dims1[0].is = (int)space_size_phy_;
	howmany_dims1[0].os = (int)space_size_phy_;

	fft_xyz_3d_r2c_forward = fftw_plan_guru_dft_r2c(rank1, dims1, howmany_rank1, howmany_dims1, iomat_for_diffusion_, step1_xyz_out, FFTW_PATIENT);
	fft_xyz_3d_c2r_backward = fftw_plan_guru_dft_c2r(rank1, dims1, howmany_rank1, howmany_dims1, step2_xyz_in_reverse, iomat_for_diffusion_, FFTW_PATIENT);
}

inline void CrysPSSolver3m::xyz_routine() const {

	constexpr size_t alignment = Alignment::value;
	const size_t Nx2 = Nx / 2;
	const size_t Ny2 = Ny / 2;
	const size_t Nz2 = Nz / 2;
	const size_t Nz_qua = ((Nz2 / 2 + 1) / (alignment / 8) + 1) * (alignment / 8);
	__assume(Nz_qua % (alignment / 8) == 0);

	fftw_execute(fft_xyz_3d_r2c_forward);

#pragma omp parallel for OMP_SCHEDULE
	for (size_t itr_x = 0; itr_x < Nx2; ++itr_x) {
		for (size_t itr_y = 0; itr_y < Ny2; ++itr_y) {
			auto base_shift = (itr_x * Ny2 + itr_y) * Nz2;
			auto src000 = step1_xyz_out + base_shift;
			auto src110 = step1_xyz_out + (((Nx2 - itr_x) % Nx2) * Ny2 + (Ny2 - itr_y) % Ny2) * Nz2;
			auto src010 = step1_xyz_out + (itr_x * Ny2 + (Ny2 - itr_y) % Ny2) * Nz2;
			auto src100 = step1_xyz_out + (((Nx2 - itr_x) % Nx2) * Ny2 + itr_y) * Nz2;
			auto dst = step2_xyz_in_reverse + base_shift;
#pragma omp simd OMP_SIMDLEN
			for (size_t itr_z = 0; itr_z < Nz_qua; ++itr_z) {
				//k seq: {0,7},{6,1},{2,5},{4,3} , or namely (0+7,6-1,2+5,4-3,4+3,2-5,6+1,0-7)
				//src seq: 000,001,010,011->000,conj@110,010,conj@100, note that the conj has already been included in k seq
				//sign seq: +000+001+010+011+100+101+110+111
				//+(000*000+111*111)->+(Re@000*(000+111)+i*Im@000*(000-111))--->Re: +(...-...), Im: +(...+...)
				//+(001*001+110*110)->+(Re@110*(110+001)+i*Im@110*(110-001))--->Re: +(...-...), Im: +(...+...)
				//+(010*010+101*101)->+(Re@010*(010+101)+i*Im@010*(010-101))--->Re: +(...-...), Im: +(...+...)
				//+(011*011+100*100)->+(Re@100*(100+011)+i*Im@100*(100-011))--->Re: +(...-...), Im: +(...+...)
				dst[itr_z][0]
					= src000[itr_z][0] * k_re[0][base_shift + itr_z] - src000[itr_z][1] * k_im[7][base_shift + itr_z]
					+ src110[itr_z][0] * k_re[6][base_shift + itr_z] - src110[itr_z][1] * k_im[1][base_shift + itr_z]
					+ src010[itr_z][0] * k_re[2][base_shift + itr_z] - src010[itr_z][1] * k_im[5][base_shift + itr_z]
					+ src100[itr_z][0] * k_re[4][base_shift + itr_z] - src100[itr_z][1] * k_im[3][base_shift + itr_z];
				dst[itr_z][1]
					= src000[itr_z][0] * k_im[0][base_shift + itr_z] + src000[itr_z][1] * k_re[7][base_shift + itr_z]
					+ src110[itr_z][0] * k_im[6][base_shift + itr_z] + src110[itr_z][1] * k_re[1][base_shift + itr_z]
					+ src010[itr_z][0] * k_im[2][base_shift + itr_z] + src010[itr_z][1] * k_re[5][base_shift + itr_z]
					+ src100[itr_z][0] * k_im[4][base_shift + itr_z] + src100[itr_z][1] * k_re[3][base_shift + itr_z];
			}
		}
	}
	fftw_execute(fft_xyz_3d_c2r_backward);
}
