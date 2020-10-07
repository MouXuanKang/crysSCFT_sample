#pragma once
#include "utilities/c_array.h"

#ifndef Pi
#define Pi 3.1415926535897932384626	
#endif // !Pi

class CrysPSSolverBase {
public:
	CrysPSSolverBase(const c_array<size_t, 3>& grid_size_log) :
		grid_size_log_(grid_size_log),
		space_size_log_(grid_size_log[0] * grid_size_log[1] * grid_size_log[2]) {}
	virtual ~CrysPSSolverBase() {}
	//set the cell parameters of the computational box:
	//{Lx, Ly, Lz, alpha, beta, gamma}
	virtual void set_cell_para(const c_array<double, 6>& cell_para) = 0;
	//set the contour step of the pseudospectral solver
	virtual void set_contour_step(double ds) = 0;
	// calculate the forward fft-->elememtwise multiplication-->backward fft triplet
	virtual void diffusion() const = 0;
	
	c_array<double, 6> get_cell_para() const { return cell_para_; }
	c_array<size_t, 3> get_grid_size_log() const { return grid_size_log_; }
	c_array<size_t, 3> get_grid_size_phy() const { return grid_size_phy_; }
	size_t get_space_size_log() const { return space_size_log_; }
	size_t get_space_size_phy() const { return space_size_phy_; }
	double get_ds() const { return ds_current_; }
	double* get_iomat_for_diffusion() const { return iomat_for_diffusion_; }

protected:
	//logical size of the computational box
	const c_array<size_t, 3> grid_size_log_;
	const size_t space_size_log_;

	//physical size of the computational box 
	//The default values are samples only.
	//These values should be changed by the ctor of derived classes, based on its corresponding symmetry.
	c_array<size_t, 3> grid_size_phy_ = { 64,64,64 };
	size_t space_size_phy_ = 262144;

	//cell parameters of the computational box
	//The last three values are reserved for a "real triclinic" computational box and not used currently.
	c_array<double, 6> cell_para_ = { 4,4,4,0,0,0 };

	//current contour step used by the solver
	double ds_current_ = 0.00;

	//input/output matrix of the solver
	//The memory must be managed by the derived class, based on its corresponding symmetry.
	double* iomat_for_diffusion_ = nullptr;
};
