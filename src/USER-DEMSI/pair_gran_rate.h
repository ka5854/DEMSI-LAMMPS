/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(gran/rate,PairGranRate)

#else

#ifndef LMP_PAIR_GRAN_RATE_H
#define LMP_PAIR_GRAN_RATE_H

#include "pair_gran_hooke_history.h"

namespace LAMMPS_NS {

class PairGranRate : public PairGranHookeHistory {
public:
  PairGranRate(class LAMMPS *);
  virtual ~PairGranRate();
  virtual void compute(int, int);
  void settings(int, char **);
  double single(int, int, int, int, double, double, double, double &);
  virtual void transfer_history(double*, double*);
  double init_one(int, int);
protected:
  void compute_rate_exverlet();
  void compute_rate_explicit();
  void compute_rate_implicit();
  void load_new_forces();

  void allocate();

  int history_ndim;

  double  bulkModulus;
  double shearModulus;
  double       shearYieldStress;
  double compressiveYieldStress;
  double  tensileFractureStress;
  double timeIntegrationFlag;
  double bulkCFL;
  
/*! math_const.h
  static const double THIRD  = 1.0/3.0;
  static const double TWOTHIRDS  = 2.0/3.0;
  static const double MY_PI  = 3.14159265358979323846; // pi
  static const double MY_2PI = 6.28318530717958647692; // 2pi
  static const double MY_3PI = 9.42477796076937971538; // 3pi
  static const double MY_4PI = 12.56637061435917295384; // 4pi
  static const double MY_PI2 = 1.57079632679489661923; // pi/2
  static const double MY_PI4 = 0.78539816339744830962; // pi/4
  static const double MY_PIS = 1.77245385090551602729; // sqrt(pi)
  static const double MY_ISPI4 = 1.12837916709551257389; // 1/sqrt(pi/4)
  static const double MY_SQRT2 = 1.41421356237309504880; // sqrt(2)
  static const double MY_CBRT2 = 1.25992104989487316476; // 2*(1/3)
*/
  const double OverLap = 1.050075135808664;   // R^\prime/R = \sqrt(2\sqrt(3) \slash \pi)
  const double LapOver = 0.9523128068639574;  // R/R^\prime = \sqrt(\pi \slash 2\sqrt(3))
  const double rho0 = 900.;  // kg/m^3
  const double MY_PI3 = 1.0471975511965976; // pi/3
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

 */
