/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(nve/sphere/demsi,FixNVESphereDemsi)

#else

#ifndef LMP_FIX_NVE_SPHERE_DEMSI_H
#define LMP_FIX_NVE_SPHERE_DEMSI_H

#include "fix_nve.h"

namespace LAMMPS_NS {

class FixNVESphereDemsi : public FixNVE {
 public:
  FixNVESphereDemsi(class LAMMPS *, int, char **);
  virtual ~FixNVESphereDemsi() {}
  void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();

  //DEMSI sets these (presumably they never change during a run;
  // if they are location/temperature dependent, they would have
  // to be made into per-particle properties)
  double ocean_drag, ocean_density;
  double atmos_drag, atmos_density;
  int drag_force_integration_flag;
  int time_integration_flag;
  double Hugoniot_Vel_Jump;

 protected:
  double inertia;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix nve/sphere disc requires 2d simulation

UNDOCUMENTED

E: Fix nve/sphere requires atom style sphere

Self-explanatory.

E: Fix nve/sphere update dipole requires atom attribute mu

An atom style with this attribute is needed.

E: Fix nve/sphere requires extended particles

This fix can only be used for particles of a finite size.

U: Fix nve/sphere dlm must be used with update dipole

The DLM algorithm can only be used in conjunction with update dipole.

*/
