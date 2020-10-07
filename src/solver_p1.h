#pragma once

#include <map>
#include <vector>
#include <iostream>
#include "fftw3.h"
#include "solver_base.h"

#include "utilities/omp_macros.h"
#include "utilities/c_array.h"
#include "utilities/assume_hint.h"
#include "utilities/alignment.h"

class CrysPSSolverP1 : public CrysPSSolverBase {
public:
	CrysPSSolverP1(const c_array<size_t, 3>& grid_size_log)
		: CrysPSSolverBase(grid_size_log) {
		set_grid();
	}

	//virtual functions
	~CrysPSSolverP1() override;
	void set_cell_para(const c_array<double, 6>& cell_para) override;
	void set_contour_step(double ds) override;
	void diffusion() const override;

private:
	constexpr static size_t alignment = Alignment::value;

	//for auto calc and record ds
	double* K_mat_current;
	std::map<double, std::vector<double*> > K_mat_collection;

	size_t Nx, Ny, Nz;
	size_t Nz_ld;
	size_t Nz_ld_ava;
	size_t space_size_phy_complex_;

	//fftw plans for diffusion
	fftw_plan xyz_3d_r2c_forward;
	fftw_plan xyz_3d_c2r_backward;

	//for diffusion 
	double* step1_xyz_out;

	void set_grid();
	void K_mat_create(double ds);
	void K_mat_destory();
	void K_mat_generator(double* K_mat, double dz, double _Lx, double _Ly, double _Lz);
	void fftw_plan_initial();
	void xyz_routine() const;
	void complex_set_zero(double* mat) const;
};

inline CrysPSSolverP1::~CrysPSSolverP1()
{
	//iomat
	_mm_free(iomat_for_diffusion_);
	//for diffusion
	_mm_free(step1_xyz_out);
	K_mat_destory();
}

inline void CrysPSSolverP1::set_cell_para(const c_array<double, 6>& cell_para)
{
	cell_para_ = { cell_para[0],cell_para[1],cell_para[2],0,0,0 };
	K_mat_destory();
	ds_current_ = 0.00;
}

inline void CrysPSSolverP1::set_contour_step(double ds)
{
	if ((ds == ds_current_) && (K_mat_collection.size() != 0)) return;
	ds_current_ = ds;
	auto target = K_mat_collection.find(ds);
	if (target != K_mat_collection.end()) K_mat_current = target->second.at(0);
	else {
		K_mat_create(ds);
		K_mat_generator(K_mat_current, ds, cell_para_[0], cell_para_[1], cell_para_[2]);
	}
}

inline void CrysPSSolverP1::diffusion() const
{
	__assume(space_size_phy_ % (alignment / 8) == 0);
	xyz_routine();
	double temp_coef = 1. / space_size_log_;
	auto iomat_for_diffusion_local = iomat_for_diffusion_;
#pragma omp parallel for simd aligned(iomat_for_diffusion_local:alignment) OMP_SIMDLEN_SCHEDULE
	for (size_t ijk = 0; ijk < space_size_phy_; ++ijk) {
		iomat_for_diffusion_local[ijk] *= temp_coef;
	}
}

inline void CrysPSSolverP1::set_grid()
{
	Nx = grid_size_log_[0];
	Ny = grid_size_log_[1];
	Nz = grid_size_log_[2];

	if (Nx <= 0 || Ny <= 0 || Nz <= 0) {
		std::cout << "ERROR: Bad Grid Size For Group Triclinic P1" << std::endl;
		exit(1);
	}
	if (Nz % (alignment / 8) != 0) {
		std::cout << "ERROR: Bad Grid Size For Group Triclinic P1, alignment cannot be guaranteed" << std::endl;
		exit(1);
	}

	grid_size_phy_ = grid_size_log_;
	space_size_phy_ = space_size_log_;
	Nz_ld = 2 * (Nz / 2 + (alignment / 8));
	Nz_ld_ava = 2 * (Nz / 2 + 1);
	space_size_phy_complex_ = Nx * Ny * Nz_ld;

	//iomat
	iomat_for_diffusion_ = (double*)_mm_malloc(space_size_phy_ * sizeof(double), alignment);
	//for diffusion
	step1_xyz_out = (double*)_mm_malloc(space_size_phy_complex_ * sizeof(double), alignment);

	//set zeros for some complex matrix, since the simd usage requires more elements to be manipulated during some operations
	//leading to the access of elements that does not have physical meanings
	complex_set_zero(step1_xyz_out);
	fftw_plan_initial();
}

inline void CrysPSSolverP1::K_mat_create(double ds)
{
	K_mat_current = (double*)_mm_malloc(space_size_phy_complex_ * sizeof(double), alignment);
	K_mat_collection.insert(std::make_pair(ds, std::vector<double*>{K_mat_current}));
}

inline void CrysPSSolverP1::K_mat_destory()
{
	for (const auto& K_ele : K_mat_collection) {
		_mm_free(K_ele.second.at(0));
	}
	K_mat_collection.clear();
	K_mat_current = nullptr;
}

inline void CrysPSSolverP1::K_mat_generator(double* K_mat, double dz, double _Lx, double _Ly, double _Lz)
{
	size_t i, j, k;
	double* kx = new double[Nx];
	double* ky = new double[Ny];
	double* kz = new double[Nz / 2 + 1];

	for (i = 0; i < (Nx + 1) / 2; i++)kx[i] = 2 * Pi * i * 1.0 / _Lx;
	for (i = (Nx + 1) / 2; i < Nx; i++)kx[i] = 2 * Pi * (Nx - i) * 1.0 / _Lx;
	for (i = 0; i < Nx; i++)kx[i] *= kx[i];

	for (j = 0; j < (Ny + 1) / 2; j++)ky[j] = 2 * Pi * j * 1.0 / _Ly;
	for (j = (Ny + 1) / 2; j < Ny; j++)ky[j] = 2 * Pi * (Ny - j) * 1.0 / _Ly;
	for (j = 0; j < Ny; j++)ky[j] *= ky[j];

	for (k = 0; k < Nz / 2 + 1; k++)kz[k] = 2 * Pi * k * 1.0 / _Lz;
	for (k = 0; k < Nz / 2 + 1; k++)kz[k] *= kz[k];

#pragma omp parallel for OMP_SCHEDULE
	for (size_t i = 0; i < Nx; ++i) {
		for (size_t j = 0; j < Ny; ++j) {
			for (size_t m = 0; m < Nz_ld_ava / 2; ++m) {
				K_mat[(Ny * i + j) * Nz_ld + 2 * m]
					= exp(-dz * (kx[i] + ky[j] + kz[m]));
				K_mat[(Ny * i + j) * Nz_ld + 2 * m + 1] = K_mat[(Ny * i + j) * Nz_ld + 2 * m];
			}
		}
	}

	delete[] kx, ky, kz;
}

inline void CrysPSSolverP1::fftw_plan_initial()
{
	int in[3] = { (int)Nx, (int)Ny, (int)Nz };
	int inembed[3] = { (int)Nx, (int)Ny, (int)Nz };
	int onembed[3] = { (int)Nx, (int)Ny, (int)Nz_ld / 2 };

	xyz_3d_r2c_forward = fftw_plan_many_dft_r2c(3, in, 1, iomat_for_diffusion_, inembed, 1, 0, (fftw_complex*)step1_xyz_out, onembed, 1, 0, FFTW_PATIENT);
	xyz_3d_c2r_backward = fftw_plan_many_dft_c2r(3, in, 1, (fftw_complex*)step1_xyz_out, onembed, 1, 0, iomat_for_diffusion_, inembed, 1, 0, FFTW_PATIENT);
}

inline void CrysPSSolverP1::xyz_routine() const {
	__assume(Nz_ld % (alignment / 8) == 0);
	fftw_execute(xyz_3d_r2c_forward);

	auto step1_xyz_out_local = step1_xyz_out;
	auto K_mat_current_local = K_mat_current;
#pragma omp parallel
	{
#pragma omp for OMP_SCHEDULE
		for (size_t itr_xy = 0; itr_xy < Nx * Ny; ++itr_xy) {
#pragma omp simd aligned(step1_xyz_out_local,K_mat_current_local:alignment) OMP_SIMDLEN
			for (size_t itr_z = 0; itr_z < Nz_ld; ++itr_z) {
				//Nz_ld is used instead of Nz_ld_ava to allow omp simd in icc
				step1_xyz_out_local[itr_xy * Nz_ld + itr_z] *= K_mat_current_local[itr_xy * Nz_ld + itr_z];
			}
		}
	}
	fftw_execute(xyz_3d_c2r_backward);
}

inline void CrysPSSolverP1::complex_set_zero(double* mat) const
{
	__assume(space_size_phy_complex_ % (alignment / 8) == 0);
#pragma omp parallel
	{
#pragma omp for simd aligned(mat:alignment) OMP_SIMDLEN_SCHEDULE
		for (size_t itr = 0; itr < space_size_phy_complex_; ++itr) {
			mat[itr] = 0;
		}
	}
}
