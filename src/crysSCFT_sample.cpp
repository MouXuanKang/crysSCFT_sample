// crysSCFT_sample.cpp : 
// This is the sample code for the paper 
// "Accelerated pseudo-spectral method of self-consistent field theory via crystallographic fast Fourier transform"
// Submitted to Macromolecules.
// In this sample, three types of diffusion equation solvers are provided:
// 1) p1 solver: assuming no symmetry, based on normal DFT from FFTW3
// 2) 3m solver: assuming that there exist three symmetry planes with each of them vertical to the others, based on normal DFT from FFTW3 with 1/8 size of the first one.
// 3) Pmmm solver: assuming that there exist three mirrors with each of them vertical to the others, based on DCTs from FFTW3 with 1/8 size of the first one.
// Three solvers are used to solve the diffusion equations of AB diblock copolymer respectively, and iterate for same steps.
// In this sample, the initial fields are read from an external file and the fields are guaranteed to have the Pmmm symmetry.
// Thus all three solvers are applicable, and give the same free energy and incompressibility at a certain iteration step.
// Since crystallographic fast Fourier transform is used in solver 2 and 3, they provide significant speedup.
// Yicheng Qiang

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <ctime>
#include <type_traits>
#include <cmath>
#include <omp.h>
#include "solver_p1.h"
#include "solver_3m.h"
#include "solver_Pmmm.h"

/// output the fields
void output_fields(
	const double* phiA, const double* phiB, const double* wA, const double* wB,
	size_t size, std::string filename
);
/// input the fields
void input_fields(
	double* phiA, double* phiB, double* wA, double* wB,
	size_t size, std::string filename
);

/// set all the elements in array to val
void array_set_vals(double* dst, double val, size_t size);

/// copy array src to dst
void array_copy(double* dst, const double* src, size_t size);

/// calculate dst[itr] *= src[itr]
void array_multip(double* dst, const double* src, size_t size);

/// calculate dst[itr] = coef*exp(index*src[itr])
void array_exp(double* dst, const double* src, double coef, double index, size_t size);

/// calculate the average of the array and return
double array_average(const double* src, size_t size);

/// <summary>
/// move a pointer forward or backward. return ptr+step if direction==1 or ptr-step if direction==-1
/// </summary>
/// <param name="ptr"> base pointer </param>
/// <param name="step"> move step </param>
/// <param name="direction"> move direction </param>
/// <returns></returns>
double* move_pointer(double* ptr, size_t step, int direction);

/// <summary>
/// calculate the lagrangian multiplier
/// </summary>
/// <param name="ksi"> lagrangian multiplier</param>
/// <param name="wA"> potential field of component A</param>
/// <param name="wB"> potential field of component B</param>
/// <param name="size"> size of the field </param>
void calc_ksi(double* ksi, const double* wA, const double* wB, size_t size, double xN);

/// <summary>
/// Integrate the propagators, q and qc, to obtain the volume fraction field 
/// </summary>
/// <param name="phi"> volume fraction field </param>
/// <param name="q"> forward propagator </param>
/// <param name="qc"> backward propagator </param>
/// <param name="f"> length of the block </param>
/// <param name="Q"> partition function </param>
/// <param name="size"> size of the field </param>
/// <param name="Ns"> number of the integration step </param>
void integrate_phi(
	double* phi, const double* q, const double* qc, double f,
	double Q, size_t size, size_t Ns
);

/// <summary>
/// calculate new potential fields
/// </summary>
/// <param name="wA"> potential field of component A </param>
/// <param name="wB"> potential field of component B </param>
/// <param name="phiA"> volume fraction field of component A </param>
/// <param name="phiB"> volume fraction field of component A </param>
/// <param name="ksi"> lagrangian multiplier </param>
/// <param name="xN"> Flory Huggins parameter </param>
/// <param name="acc"> acceptance for the iteration </param>
/// <param name="size"> size of the field </param>
void calc_new_fields(
	double* wA, double* wB, const double* phiA, const double* phiB, const double* ksi,
	double xN, double acc, size_t size
);

/// <summary>
/// solve a 2D modified diffusion equation
/// </summary>
/// <param name="q"> propagator </param>
/// <param name="init"> inital state </param>
/// <param name="w"> potential field </param>
/// <param name="k"> eigen values of Laplacian </param>
/// <param name="f"> length of the block </param>
/// <param name="Nx"> spatial points on x direction </param>
/// <param name="Ny"> spatial points on y direction </param>
/// <param name="Ns"> number of the integration step </param>
/// <param name="direction"> direction of the propagator </param>
void solve_diffusion_equation(
	double* q, const double* init, const double* wds,
	const CrysPSSolverBase& solver, size_t Ns, int direction, std::clock_t& time
);

/// <summary>
/// calculate and print out the free energy
/// </summary>
/// <param name="itr"> iteration </param>
/// <param name="Q"> partition function </param>
/// <param name="xN"> Flory-Huggins parameter </param>
/// <param name="wA"> potential field of component A </param>
/// <param name="wB"> potential field of component A </param>
/// <param name="phiA"> volume fraction distribution of component A </param>
/// <param name="phiB"> volume fraction distribution of component B </param>
/// <param name="size"> size of the array </param>
void print_energy(
	size_t itr, double Q, double xN,
	const double* wA, const double* wB, const double* phiA, const double* phiB, const double* ksi,
	size_t size
);

/// <summary>
/// solve SCFT equations for AB diblock copolymer. 
/// </summary>
/// <typeparam name="InitFuncType"> Infered </typeparam>
/// <param name="Nx"> spatial points along x direction </param>
/// <param name="Ny"> spatial points along y direction </param>
/// <param name="Nz"> spatial points along z direction </param>
/// <param name="fA"> volume fraction of component A </param>
/// <param name="ds"> step along the chain contour </param>
/// <param name="xN"> Flory-Huggins parameter </param>
/// <param name="Lx"> periods in x direction </param>
/// <param name="Ly"> periods in y direction </param>
/// <param name="Lz"> periods in z direction </param>
/// <param name="acceptance"> acceptance of the fields in simple iteration </param>
/// <param name="Nstep"> step to calculate </param>
/// <param name="solvername"> name of the solver used, used for print infomations </param>
/// <param name="solver"> the solver for diffusion equations </param>
/// <param name="init_func"> function for initialization of the fields. It takes two double* as input </param>
/// <returns>time consumed by the triplet</returns>
template<typename InitFuncType>
std::clock_t diblock_scft_simple_iterate(
	size_t Nx, size_t Ny, size_t Nz,
	double fA, double ds, double xN,
	double Lx, double Ly, double Lz,
	double acceptance, size_t Nstep,
	std::string solvername,
	CrysPSSolverBase& solver,
	InitFuncType init_func,
	typename std::enable_if<std::is_invocable_r_v<void, InitFuncType, double*, double*>, void**>::type = nullptr
);

int main()
{
	omp_set_num_threads(1);

	//set parameters
	constexpr size_t Nx = 64;
	constexpr size_t Ny = 64;
	constexpr size_t Nz = 64;
	double fA = 0.2;
	double ds = 0.01;
	double xN = 30;
	double Lx = 5;
	double Ly = 5;
	double Lz = 5;
	double acceptance = 0.05;
	size_t Nstep = 100;

	auto simple_test_func = [&](auto name, auto& solver, auto initfunc) ->std::clock_t {
		return diblock_scft_simple_iterate(Nx, Ny, Nz, fA, ds, xN, Lx, Ly, Lz, acceptance, Nstep, name, solver, initfunc);
	};

	constexpr size_t alignment = Alignment::value;
	constexpr size_t size = Nx * Ny * Nz;

	//allocate memorys
	double* wAinit = (double*)_mm_malloc(size * sizeof(double), alignment);
	double* wBinit = (double*)_mm_malloc(size * sizeof(double), alignment);
	double* dummy = (double*)_mm_malloc(size * sizeof(double), alignment);
	//read the initial file
	input_fields(dummy, dummy, wAinit, wBinit, size, "BCC_fields_test.txt");

	//normal solver, based on DFT from FFTW3
	std::string name_normal("SolverNormal");
	CrysPSSolverP1 solver_normal({ Nx,Ny,Nz });
	//normal solver use whole fields in SCFT
	auto initfunc_normal =
		[size = size, wAinit = (const double*)wAinit, wBinit = (const double*)wBinit](double* wA, double* wB) ->void
	{
		for (size_t itr = 0; itr < size; ++itr) {
			wA[itr] = wAinit[itr];
			wB[itr] = wBinit[itr];
		}
	};
	auto time_normal = simple_test_func(name_normal, solver_normal, initfunc_normal);

	//generalized solver for symmetry with three symmetry planes perpendicular to each other
	//based on DFT of 1/8 size from FFTW3
	//The second parameter for ctor is the translational part of the symmetry plane.
	//The translational part of the symmetry plane vertical to z direction comes first.
	std::string name_3m("Solver3m");
	CrysPSSolver3m solver_3m({ Nx,Ny,Nz }, { 0,0,0,0,0,0,0,0,0 });
	//3m solver use elements with even indexes from the whole fields
	auto initfunc_3m =
		[
			Nx2 = Nx / 2, Ny2 = Ny / 2, Nz2 = Nz / 2,
			wAinit = (const double*)wAinit, wBinit = (const double*)wBinit
		](double* wA, double* wB) ->void
	{
		for (size_t itr_x = 0; itr_x < Nx2; ++itr_x) {
			for (size_t itr_y = 0; itr_y < Ny2; ++itr_y) {
				for (size_t itr_z = 0; itr_z < Nz2; ++itr_z) {
					auto index_src = (2 * itr_x * Ny2 * 2 + 2 * itr_y) * Nz2 * 2 + 2 * itr_z;
					auto index_dst = (itr_x * Ny2 + itr_y) * Nz2 + itr_z;
					wA[index_dst] = wAinit[index_src];
					wB[index_dst] = wBinit[index_src];
				}
			}
		}
	};
	auto time_3m = simple_test_func(name_3m, solver_3m, initfunc_3m);

	//solver for Pmmm symmetry, based on DCT of 1/8 size from FFTW3
	std::string name_Pmmm_dct("SolverPmmmDCT");
	CrysPSSolverPmmm solver_Pmmm_dct({ Nx,Ny,Nz });
	//Pmmm solver use first 1/8 of the whole fields
	auto initfunc_Pmmm_dct =
		[
			Nx2 = Nx / 2, Ny2 = Ny / 2, Nz2 = Nz / 2,
			wAinit = (const double*)wAinit, wBinit = (const double*)wBinit
		](double* wA, double* wB) ->void
	{
		for (size_t itr_x = 0; itr_x < Nx2; ++itr_x) {
			for (size_t itr_y = 0; itr_y < Ny2; ++itr_y) {
				for (size_t itr_z = 0; itr_z < Nz2; ++itr_z) {
					auto index_src = (itr_x * Ny2 * 2 + itr_y) * Nz2 * 2 + itr_z;
					auto index_dst = (itr_x * Ny2 + itr_y) * Nz2 + itr_z;
					wA[index_dst] = wAinit[index_src];
					wB[index_dst] = wBinit[index_src];
				}
			}
		}
	};
	auto time_Pmmm = simple_test_func(name_Pmmm_dct, solver_Pmmm_dct, initfunc_Pmmm_dct);

	std::cout<<"The estimated speedup of Solver3m is "<<(double)time_normal/time_3m<<std::endl;
	std::cout<<"The estimated speedup of SolverPmmmDCT is "<<(double)time_normal/time_Pmmm<<std::endl;

	//deallocate memory
	_mm_free(wAinit);
	_mm_free(wBinit);
	_mm_free(dummy);
	return 0;
}

void output_fields(
	const double* phiA, const double* phiB, const double* wA, const double* wB,
	size_t size, std::string filename
)
{
	std::ofstream ofs(filename);

	for (size_t itr = 0; itr < size; ++itr) {
		ofs << phiA[itr] << " " << phiB[itr] << " " << wA[itr] << " " << wB[itr] << "\n";
	}
	ofs.close();
}

void input_fields(double* phiA, double* phiB, double* wA, double* wB, size_t size, std::string filename)
{
	std::ifstream ifs(filename);

	for (size_t itr = 0; itr < size; ++itr) {
		ifs >> phiA[itr] >> phiB[itr] >> wA[itr] >> wB[itr];
	}
	ifs.close();
}

void array_set_vals(double* dst, double val, size_t size)
{
	for (size_t itr = 0; itr < size; ++itr) {
		dst[itr] = val;
	}
}

void array_copy(double* dst, const double* src, size_t size)
{
	for (size_t itr = 0; itr < size; ++itr) {
		dst[itr] = src[itr];
	}
}

void array_multip(double* dst, const double* src, size_t size)
{
	for (size_t itr = 0; itr < size; ++itr) {
		dst[itr] *= src[itr];
	}
}

void array_exp(
	double* dst, const double* src, double coef, double index, size_t size
)
{
	for (size_t itr = 0; itr < size; ++itr) {
		dst[itr] = coef * std::exp(index * src[itr]);
	}
}

double array_average(const double* src, size_t size)
{
	double sum = 0.;
	for (size_t itr = 0; itr < size; ++itr) {
		sum += src[itr];
	}
	return sum / size;
}

double* move_pointer(double* ptr, size_t step, int direction)
{
	if (direction == 1) return ptr + step;
	if (direction == -1) return ptr - step;
	std::cerr << "Wrong direction" << std::endl;
	exit(1);
}

void calc_ksi(double* ksi, const double* wA, const double* wB, size_t size, double xN)
{
	for (size_t itr = 0; itr < size; ++itr) {
		ksi[itr] = 0.5 * (wA[itr] + wB[itr] - xN);
	}
}

void integrate_phi(
	double* phi, const double* q, const double* qc,
	double f, double Q, size_t size, size_t Ns
)
{
	for (size_t itr_s = 0; itr_s <= Ns; ++itr_s) {
		double factor = itr_s == 0 ? 0.5 : (itr_s == Ns ? 0.5 : 1);
		factor *= f / (Ns * Q);
		for (size_t itr = 0; itr < size; ++itr) {
			phi[itr] += factor * q[itr_s * size + itr] * qc[itr_s * size + itr];
		}
	}
}

void calc_new_fields(
	double* wA, double* wB, const double* phiA, const double* phiB,
	const double* ksi, double xN, double acc, size_t size
)
{
	for (size_t itr = 0; itr < size; ++itr) {
		wA[itr] += acc * (xN * phiB[itr] + ksi[itr] - wA[itr]);
		wB[itr] += acc * (xN * phiA[itr] + ksi[itr] - wB[itr]);
	}
}

void solve_diffusion_equation(double* q, const double* init, const double* wds, const CrysPSSolverBase& solver, size_t Ns, int direction, std::clock_t& time)
{
	double* solver_space = solver.get_iomat_for_diffusion();
	double* current_q;
	size_t size = solver.get_space_size_phy();
	if (direction == 1) current_q = q;
	else if (direction == -1) current_q = q + size * Ns;
	else {
		std::cerr << "Wrong direction" << std::endl;
		exit(-1);
	}

	array_copy(solver_space, init, size);
	array_copy(current_q, init, size);
	for (size_t itr_s = 0; itr_s < Ns; ++itr_s) {
		array_multip(solver_space, wds, size);
		std::clock_t temp = std::clock();
		solver.diffusion();
		time += std::clock() - temp;//record the time consumed by the triplet
		array_multip(solver_space, wds, size);
		current_q = move_pointer(current_q, size, direction);
		array_copy(current_q, solver_space, size);
	}
}

void print_energy(
	size_t itr, double Q, double xN,
	const double* wA, const double* wB, const double* phiA, const double* phiB, const double* ksi,
	size_t size
)
{
	double fe_entropic_0 = -std::log(Q);
	double fe_entropic_1 = 0;
	double fe_enthalpic = 0;
	double fe_ksi = 0;
	double maxincomp = std::fabs(1 - phiA[0] - phiB[0]);
	for (size_t itr = 0; itr < size; ++itr) {
		fe_entropic_1 += -wA[itr] * phiA[itr] - wB[itr] * phiB[itr];
		fe_enthalpic += xN * phiA[itr] * phiB[itr];
		double temp = 1 - phiA[itr] - phiB[itr];
		fe_ksi += -ksi[itr] * temp;
		maxincomp = std::fabs(temp) > maxincomp ? std::fabs(temp) : maxincomp;
	}
	fe_entropic_1 /= size;
	fe_enthalpic /= size;
	fe_ksi /= size;
	std::cout << std::setprecision(10);
	std::cout 
	<< "iter " << itr << ": FreeEnergy = " << fe_entropic_0 + fe_entropic_1 + fe_enthalpic + fe_ksi
	<< " ( "<<fe_enthalpic <<" + "<<fe_entropic_0<<" + "<<fe_entropic_1<<" + "<<fe_ksi<<" ) "
	<< " , MaxIncompressibility = " << maxincomp << std::endl;
}

template<typename InitFuncType>
std::clock_t diblock_scft_simple_iterate(
	size_t Nx, size_t Ny, size_t Nz,
	double fA, double ds, double xN,
	double Lx, double Ly, double Lz,
	double acceptance, size_t Nstep,
	std::string solvername,
	CrysPSSolverBase& solver,
	InitFuncType init_func,
	typename std::enable_if<std::is_invocable_r_v<void, InitFuncType, double*, double*>, void**>::type
)
{
	constexpr size_t alignment = Alignment::value;

	//derived parameters
	size_t size = solver.get_space_size_phy();
	double fB = 1 - fA;
	size_t NsA = fA / ds;
	size_t NsB = fB / ds;

	//allocate memorys
	double* wA = (double*)_mm_malloc(size * sizeof(double), alignment);
	double* wB = (double*)_mm_malloc(size * sizeof(double), alignment);
	double* wAds = (double*)_mm_malloc(size * sizeof(double), alignment);
	double* wBds = (double*)_mm_malloc(size * sizeof(double), alignment);
	double* phiA = (double*)_mm_malloc(size * sizeof(double), alignment);
	double* phiB = (double*)_mm_malloc(size * sizeof(double), alignment);
	double* ksi = (double*)_mm_malloc(size * sizeof(double), alignment);
	double* k = (double*)_mm_malloc(size * sizeof(double), alignment);
	double* init = (double*)_mm_malloc(size * sizeof(double), alignment);

	double* qA = (double*)_mm_malloc(size * (NsA + 1) * sizeof(double), alignment);
	double* qcA = (double*)_mm_malloc(size * (NsA + 1) * sizeof(double), alignment);
	double* qB = (double*)_mm_malloc(size * (NsB + 1) * sizeof(double), alignment);
	double* qcB = (double*)_mm_malloc(size * (NsB + 1) * sizeof(double), alignment);

	//initialization
	init_func(wA, wB);

	//calculate the eigenvalue
	solver.set_cell_para({ Lx,Ly,Lz });
	solver.set_contour_step(ds);

	std::cout << "******************************************************************" << std::endl;
	std::cout << "Warm up the solver " << solvername << "..." << std::endl;
	solver.diffusion();

	std::cout << "Start solving with solver " << solvername << "..." << std::endl;
	std::clock_t time = 0;
	//main loop
	for (size_t itr = 0; itr < Nstep; ++itr) {
		//calculate the lagrangian Multiplier
		calc_ksi(ksi, wA, wB, size, xN);
		array_exp(wAds, wA, 1., -ds / 2, size);
		array_exp(wBds, wB, 1., -ds / 2, size);

		//solve diffusion equation
		array_set_vals(init, 1, size);
		solve_diffusion_equation(qA, init, wAds, solver, NsA, 1, time);
		solve_diffusion_equation(qcB, init, wBds, solver, NsB, -1, time);

		array_copy(init, qA + NsA * size, size);
		solve_diffusion_equation(qB, init, wBds, solver, NsB, 1, time);

		array_copy(init, qcB, size);
		solve_diffusion_equation(qcA, init, wAds, solver, NsA, -1, time);

		//calculate the partition function
		double Q = array_average(qcA, size);

		//calculate the volume fraction fields
		array_set_vals(phiA, 0, size);
		array_set_vals(phiB, 0, size);
		integrate_phi(phiA, qA, qcA, fA, Q, size, NsA);
		integrate_phi(phiB, qB, qcB, fB, Q, size, NsB);

		if (itr % 10 == 0 || itr == (Nstep - 1))
			print_energy(itr, Q, xN, wA, wB, phiA, phiB, ksi, size);

		//recalculate potential field
		calc_new_fields(wA, wB, phiA, phiB, ksi, xN, acceptance, size);

	}

	//output fields
	output_fields(phiA, phiB, wA, wB, size, solvername + "_output.txt");
	std::cout << time << " clocks are consumed by triplets , with CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << std::endl;
	std::cout << "******************************************************************" << std::endl;

	//deallocate memory
	_mm_free(wA);
	_mm_free(wB);
	_mm_free(wAds);
	_mm_free(wBds);
	_mm_free(phiA);
	_mm_free(phiB);
	_mm_free(ksi);
	_mm_free(k);
	_mm_free(init);
	_mm_free(qA);
	_mm_free(qcA);
	_mm_free(qB);
	_mm_free(qcB);

	return time;
}
