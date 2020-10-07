# Code Sample for Manuscript "Accelerated pseudo-spectral method of self-consistent field theory via crystallographic fast Fourier transform"
## Introduction
Pseudo-spectral method is one of the most popular numerical methods for the self-consistent field theory (SCFT) of flexible block copolymer systems. We suggest that by introducing crystallographic fast Fourier transform (FFT) into the pseudo-spectral method, the SCFT calculation of ordered phases can be significantly accelerated, compared to the version that use normal FFT.
In this sample, three solvers that implement the most general algorithms described in the manuscript "Accelerated pseudo-spectral method of self-consistent field theory via crystallographic fast Fourier transform" are provided, along with a simple SCFT program for testing.
## Solvers
Three solvers are provided:

1. SolverP1: This solver uses normal FFT from FFTW3 to calculate the forward FFT->element-wise multiplication->backward FFT triplet. It assumes no symmetry and can be applied to any ordered morphologies.
2. Solver3m: This solver implements the *2x2y2z* algorithm, based on normal FFT from FFTW3, but with only 1/8 the required size. It is applicable to morphologies with three symmetry planes vertical to each other.
3. SolverPmmmDCT: This solver uses DCTs from FFTW3 with 1/8 the required size. It is applicable to morphologies with three mirrors vertical to each other, namely, whose space group is supergroup of *Pmmm*.

## SCFT Program
A simple SCFT program for AB diblock copolymer is provided to illustrate the speedup brought by crystallographic FFT. In the sample, three solvers are used to solve the same system and iterate for same steps.
## Input Fields
A file that contains fields of a BCC morphology is provided as input. Since the space group of BCC is a supergroup of *Pmmm*, all three solvers are applicable.
## Language and Prerequisites
This sample is written in C++ and requires:

- C++17
- OpenMP
- FFTW3  
- cmake3

Our test environment is icc 19.1.1.217 (gcc version 9.3.1 compatibility) ,cmake3 3.17.3 and FFTW 3.3.3
## Example Output
The sample code gives following output in our test environment with Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
>\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
>Warm up the solver SolverNormal...  
>Start solving with solver SolverNormal...  
>iter 0: FreeEnergy = 4.320637066 ( 2.453674053 + 6.733124105 + -4.835926291 + -0.03023480109 )  , MaxIncompressibility = 0.1144624876  
>iter 10: FreeEnergy = 4.321678201 ( 2.429307501 + 6.745071219 + -4.838205853 + -0.01449466577 )  , MaxIncompressibility = 0.08934535521  
>iter 20: FreeEnergy = 4.32173588 ( 2.407735525 + 6.743027443 + -4.824793041 + -0.004234047856 )  , MaxIncompressibility = 0.05567596753  
>iter 30: FreeEnergy = 4.321537131 ( 2.392201545 + 6.734250717 + -4.80573282 + 0.0008176893329 )  , MaxIncompressibility = 0.02377007076  
>iter 40: FreeEnergy = 4.321422122 ( 2.382980666 + 6.723971993 + -4.787744867 + 0.002214331556 )  , MaxIncompressibility = 0.005403012939  
>iter 50: FreeEnergy = 4.321385372 ( 2.37876723 + 6.715139291 + -4.774130798 + 0.00160964938 )  , MaxIncompressibility = 0.01527812229  
>iter 60: FreeEnergy = 4.321382118 ( 2.377706468 + 6.708838696 + -4.76557548 + 0.0004124345386 )  , MaxIncompressibility = 0.02095243993  
>iter 70: FreeEnergy = 4.321396136 ( 2.37816057 + 6.704989186 + -4.761274629 + -0.0004789918917 )  , MaxIncompressibility = 0.0197853129  
>iter 80: FreeEnergy = 4.321411138 ( 2.379050362 + 6.70298549 + -4.759861236 + -0.0007634785236 )  , MaxIncompressibility = 0.01454924311  
>iter 90: FreeEnergy = 4.321415595 ( 2.379831296 + 6.702136473 + -4.759994504 + -0.0005576708798 )  , MaxIncompressibility = 0.007854356484  
>iter 99: FreeEnergy = 4.321412529 ( 2.380282663 + 6.701895419 + -4.760588368 + -0.000177184309 )  , MaxIncompressibility = 0.002753010438  
>67150000 clocks are consumed by triplets , with CLOCKS_PER_SEC = 1000000  
>\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
>\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
>Warm up the solver Solver3m...  
>Start solving with solver Solver3m...  
>iter 0: FreeEnergy = 4.320637066 ( 2.453674053 + 6.733124105 + -4.835926291 + -0.03023480109 )  , MaxIncompressibility = 0.1144624876  
>iter 10: FreeEnergy = 4.321678201 ( 2.429307501 + 6.745071219 + -4.838205853 + -0.01449466577 )  , MaxIncompressibility = 0.08934535521  
>iter 20: FreeEnergy = 4.32173588 ( 2.407735525 + 6.743027443 + -4.824793041 + -0.004234047856 )  , MaxIncompressibility = 0.05567596753  
>iter 30: FreeEnergy = 4.321537131 ( 2.392201545 + 6.734250717 + -4.80573282 + 0.0008176893329 )  , MaxIncompressibility = 0.02377007076  
>iter 40: FreeEnergy = 4.321422122 ( 2.382980666 + 6.723971993 + -4.787744867 + 0.002214331556 )  , MaxIncompressibility = 0.005403012939  
>iter 50: FreeEnergy = 4.321385372 ( 2.37876723 + 6.715139291 + -4.774130798 + 0.00160964938 )  , MaxIncompressibility = 0.01527812229  
>iter 60: FreeEnergy = 4.321382118 ( 2.377706468 + 6.708838696 + -4.76557548 + 0.0004124345386 )  , MaxIncompressibility = 0.02095243993  
>iter 70: FreeEnergy = 4.321396136 ( 2.37816057 + 6.704989186 + -4.761274629 + -0.0004789918917 )  , MaxIncompressibility = 0.0197853129  
>iter 80: FreeEnergy = 4.321411138 ( 2.379050362 + 6.70298549 + -4.759861236 + -0.0007634785236 )  , MaxIncompressibility = 0.01454924311  
>iter 90: FreeEnergy = 4.321415595 ( 2.379831296 + 6.702136473 + -4.759994504 + -0.0005576708798 )  , MaxIncompressibility = 0.007854356484  
>iter 99: FreeEnergy = 4.321412529 ( 2.380282663 + 6.701895419 + -4.760588368 + -0.000177184309 )  , MaxIncompressibility = 0.002753010438  
>11430000 clocks are consumed by triplets , with CLOCKS_PER_SEC = 1000000  
>\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
>\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
>Warm up the solver SolverPmmmDCT...  
>Start solving with solver SolverPmmmDCT...  
>iter 0: FreeEnergy = 4.320637066 ( 2.453674053 + 6.733124105 + -4.835926291 + -0.03023480109 )  , MaxIncompressibility = 0.1144624876  
>iter 10: FreeEnergy = 4.321678201 ( 2.429307501 + 6.745071219 + -4.838205853 + -0.01449466577 )  , MaxIncompressibility = 0.08934535521  
>iter 20: FreeEnergy = 4.32173588 ( 2.407735525 + 6.743027443 + -4.824793041 + -0.004234047856 )  , MaxIncompressibility = 0.05567596753  
>iter 30: FreeEnergy = 4.321537131 ( 2.392201545 + 6.734250717 + -4.80573282 + 0.0008176893329 )  , MaxIncompressibility = 0.02377007076  
>iter 40: FreeEnergy = 4.321422122 ( 2.382980666 + 6.723971993 + -4.787744867 + 0.002214331556 )  , MaxIncompressibility = 0.005403012939  
>iter 50: FreeEnergy = 4.321385372 ( 2.37876723 + 6.715139291 + -4.774130798 + 0.00160964938 )  , MaxIncompressibility = 0.01527812229  
>iter 60: FreeEnergy = 4.321382118 ( 2.377706468 + 6.708838696 + -4.76557548 + 0.0004124345386 )  , MaxIncompressibility = 0.02095243993  
>iter 70: FreeEnergy = 4.321396136 ( 2.37816057 + 6.704989186 + -4.761274629 + -0.0004789918917 )  , MaxIncompressibility = 0.0197853129  
>iter 80: FreeEnergy = 4.321411138 ( 2.379050362 + 6.70298549 + -4.759861236 + -0.0007634785236 )  , MaxIncompressibility = 0.01454924311  
>iter 90: FreeEnergy = 4.321415595 ( 2.379831296 + 6.702136473 + -4.759994504 + -0.0005576708798 )  , MaxIncompressibility = 0.007854356484  
>iter 99: FreeEnergy = 4.321412529 ( 2.380282663 + 6.701895419 + -4.760588368 + -0.000177184309 )  , MaxIncompressibility = 0.002753010438  
>17530000 clocks are consumed by triplets , with CLOCKS_PER_SEC = 1000000  
>\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
>The estimated speedup of Solver3m is 5.874890639  
>The estimated speedup of SolverPmmmDCT is 3.830576155  