#pragma once
#include <map>
#include <iostream>

#include "solver_base.h"
#include "solver_utilities.h"

#include "utilities/omp_macros.h"
#include "utilities/c_array.h"
#include "utilities/assume_hint.h"
#include "utilities/alignment.h"

class CrysPSSolverPmmm : public CrysPSSolverBase {

public:
	CrysPSSolverPmmm(const c_array<size_t, 3>& grid_size_log)
		: CrysPSSolverBase(grid_size_log) {
		set_grid();
	}
	//virtual functions
	inline ~CrysPSSolverPmmm() override;
	inline void set_cell_para(const c_array<double, 6>& cell_para) override;
	inline void set_contour_step(double ds) override;
	inline void diffusion() const override;

private:
	constexpr static size_t alignment = Alignment::value;

	//for auto calc and record ds
	double* K_mat_current;
	std::map<double, std::vector<double*> > K_mat_collection;

	size_t Nx;//= NN;
	size_t Ny;//= NN;
	size_t Nz;//= NN;
	size_t logical_size;//= Nx * Ny * Nz;

	//fftw plans for diffusion
	fftw_plan dct_xyz_3d_r2c_forward;
	fftw_plan dct_xyz_3d_c2r_backward;
	//for diffusion 
	double* step1_xyz_out;// = (double*)fftw_malloc(sizeof(double)*(space_size_phy_));
	double* step2_xyz_in_reverse;// = (double*)fftw_malloc(sizeof(double)*(space_size_phy_));

	inline void set_grid();
	inline void K_mat_create(double ds);
	inline void K_mat_destory();

	inline void K_mat_generator(double* K_mat, double dz, double L);
	inline void fftw_plan_initial();
	inline void xyz_routine() const;
};

inline CrysPSSolverPmmm::~CrysPSSolverPmmm()
{
	//iomat
	destroy_matrix(&iomat_for_diffusion_);

	//for diffusion 
	destroy_matrix(&step1_xyz_out);
	destroy_matrix(&step2_xyz_in_reverse);

	K_mat_destory();
}

inline void CrysPSSolverPmmm::set_grid()
{
	Nx = grid_size_log_[0];
	Ny = grid_size_log_[1];
	Nz = grid_size_log_[2];
	if (Nx % 2 || Ny % 2 || Nz % 2) {
		std::cout << "ERROR: Grid Size Must Be Even For Group Pmmm" << std::endl;
		exit(1);
	}
	if (Nx <= 0 || Ny <= 0 || Nz <= 0) {
		std::cout << "ERROR: Bad Grid Size For Group Pmmm" << std::endl;
		exit(1);
	}
	if ((Nz / 2) % (alignment / 8) != 0) {
		std::cout << "ERROR: Bad Grid Size For Group Pmmm, Alignment Cannot Be Guaranteed" << std::endl;
		exit(1);
	}

	grid_size_phy_ = { Nx / 2 ,Ny / 2 ,Nz / 2 };
	space_size_phy_ = Nx * Ny * Nz / 8;

	logical_size = Nx * Ny * Nz;

	//iomat
	make_matrix(&iomat_for_diffusion_, space_size_phy_);

	//for diffusion
	make_matrix(&step1_xyz_out, space_size_phy_);
	make_matrix(&step2_xyz_in_reverse, space_size_phy_);
	fftw_plan_initial();
}

inline void CrysPSSolverPmmm::set_cell_para(const c_array<double, 6>& cell_para)
{
	if (cell_para[0] == cell_para_[0] && cell_para[1] == cell_para_[1] && cell_para[2] == cell_para_[2] && cell_para[3] == 0 && cell_para[4] == 0 && cell_para[5] == 0) return;
	cell_para_ = { cell_para[0],cell_para[0],cell_para[0],0,0,0 };
	K_mat_destory();
	ds_current_ = 0.00;
}

inline void CrysPSSolverPmmm::set_contour_step(double ds)
{
	if (K_mat_collection.size() > 0 && ds == ds_current_) {
		return;
	}

	ds_current_ = ds;

	auto target = K_mat_collection.find(ds);

	if (target != K_mat_collection.end()) {
		K_mat_current = target->second.at(0);
	}
	else {
		K_mat_create(ds);
		K_mat_generator(K_mat_current, ds, cell_para_[0]);
	}
}

inline void CrysPSSolverPmmm::diffusion()const
{
	xyz_routine();
	double factor = 1. / logical_size;
	auto iomat_for_diffusion_local = iomat_for_diffusion_;
	size_t Nxy_outer = (Nx * Ny) / 4;
	size_t Nz_inner = Nz / 2;
	__assume(Nz_inner % (alignment / 8) == 0);
#pragma omp parallel
	{
#pragma omp for OMP_SCHEDULE
		for (size_t itr_xy = 0; itr_xy < Nxy_outer; ++itr_xy) {
#pragma omp simd aligned(iomat_for_diffusion_local:alignment) OMP_SIMDLEN
			for (size_t itr_z = 0; itr_z < Nz_inner; ++itr_z) {
				iomat_for_diffusion_local[itr_xy * Nz_inner + itr_z] *= factor;
			}
		}
	}
}

inline void CrysPSSolverPmmm::K_mat_create(double ds)
{
	make_matrix(&K_mat_current, space_size_phy_);
	K_mat_collection.insert(std::make_pair(ds, std::vector<double*>{K_mat_current}));
}

inline void CrysPSSolverPmmm::K_mat_destory()
{
	for (auto& K_ele : K_mat_collection) {
		destroy_matrix(&(K_ele.second.at(0)));
	}
	K_mat_collection.clear();
}

inline void CrysPSSolverPmmm::K_mat_generator(double* K_mat, double dz, double L)
{
	size_t i, j, k, ijk;
	double* kx = new double[Nx / 2];
	double* ky = new double[Ny / 2];
	double* kz = new double[Nz / 2];

	for (i = 0; i < Nx / 2; i++)kx[i] = 2 * Pi * i * 1.0 / L;
	for (i = 0; i < Nx / 2; i++)kx[i] *= kx[i];

	for (j = 0; j < Ny / 2; j++)ky[j] = 2 * Pi * j * 1.0 / L;
	for (j = 0; j < Ny / 2; j++)ky[j] *= ky[j];

	for (k = 0; k < Nz / 2; k++)kz[k] = 2 * Pi * k * 1.0 / L;
	for (k = 0; k < Nz / 2; k++)kz[k] *= kz[k];

	ijk = 0;
	for (i = 0; i < Nx / 2; ++i) {
		for (j = 0; j < Ny / 2; ++j) {
			for (k = 0; k < Nz / 2; ++k) {
				K_mat[ijk] = exp(-(kx[i] + ky[j] + kz[k]) * dz);
				++ijk;
			}
		}
	}

	delete[] kx, ky, kz;
}

inline void CrysPSSolverPmmm::fftw_plan_initial()
{
	dct_xyz_3d_r2c_forward = fftw_plan_r2r_3d((int)Nx / 2, (int)Ny / 2, (int)Nz / 2, iomat_for_diffusion_, step1_xyz_out, FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10, FFTW_PATIENT);
	dct_xyz_3d_c2r_backward = fftw_plan_r2r_3d((int)Nx / 2, (int)Ny / 2, (int)Nz / 2, step2_xyz_in_reverse, iomat_for_diffusion_, FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01, FFTW_PATIENT);
}

inline void CrysPSSolverPmmm::xyz_routine() const {
	fftw_execute(dct_xyz_3d_r2c_forward);

	auto step2_xyz_in_reverse_local = step2_xyz_in_reverse;
	auto step1_xyz_out_local = step1_xyz_out;
	auto K_mat_current_local = K_mat_current;
	size_t Nxy_outer = (Nx * Ny) / 4;
	size_t Nz_inner = Nz / 2;
	__assume(Nz_inner % (alignment / 8) == 0);
#pragma omp parallel
	{
#pragma omp for OMP_SCHEDULE
		for (size_t itr_xy = 0; itr_xy < Nxy_outer; ++itr_xy) {
#pragma omp simd aligned(step2_xyz_in_reverse_local,step1_xyz_out_local,K_mat_current_local:alignment) OMP_SIMDLEN
			for (size_t itr_z = 0; itr_z < Nz_inner; ++itr_z) {
				step2_xyz_in_reverse_local[itr_xy * Nz_inner + itr_z]
					= step1_xyz_out_local[itr_xy * Nz_inner + itr_z] * K_mat_current_local[itr_xy * Nz_inner + itr_z];
			}
		}
	}
	fftw_execute(dct_xyz_3d_c2r_backward);
}
