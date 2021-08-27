/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdio>
#include <cstring>
#include "fix_nve_sphere_demsi.h"
#include "atom.h"
#include "domain.h"
#include "atom_vec.h"
#include "update.h"
#include "respa.h"
#include "force.h"
#include "error.h"
#include "math_extra.h"
#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathExtra;

enum{NONE,DIPOLE};
enum{NODLM,DLM};

/* ---------------------------------------------------------------------- */

FixNVESphereDemsi::FixNVESphereDemsi(LAMMPS *lmp, int narg, char **arg) :
  FixNVE(lmp, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal fix nve/sphere/demsi command");

  time_integrate = 1;

  // process extra keywords
  // inertia = moment of inertia prefactor for sphere or disc
  inertia = 0.5;

  ocean_density = ocean_drag = 0;

  timeIntegrationFlag = 0;
  bulkCFL = 1.;
  bulkModulus = 1.;
  shearModulus = 1.;

  if (domain->dimension != 2)
    error->all(FLERR,"Fix nve/sphere demsi requires 2d simulation");
  if (!atom->demsi_flag)
    error->all(FLERR,"Fix nve/sphere requires atom style demsi");
}

/* ---------------------------------------------------------------------- */

void FixNVESphereDemsi::init()
{
  FixNVE::init();

  // check that all particles are finite-size spheres
  // no point particles allowed

  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      if (radius[i] == 0.0)
        error->one(FLERR,"Fix nve/sphere requires extended particles");
}

/* ---------------------------------------------------------------------- */

void FixNVESphereDemsi::initial_integrate(int /*vflag*/)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  double *ice_area = atom->ice_area;
  double *coriolis = atom->coriolis;
  double **ocean_vel = atom->ocean_vel;
  double **bvector = atom->bvector;
  double **forcing = atom->forcing;
  double **vn = atom->vn; // adding vn
  double *mean_thickness = atom->mean_thickness;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // set timestep here since dt may have changed or come via rRESPA
  double dtv = update->dt;
  double dtf = 0.5*dtv;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {

        vn[i][0] =     v[i][0];
        vn[i][1] =     v[i][1];
        vn[i][2] = omega[i][2];

//      x[i][0] += dtf * v[i][0];  // half step with time(n) vel
//      x[i][1] += dtf * v[i][1];

        // half step acceleration with time(n) forces
        double dtm = dtf/rmass[i];
        double radi = radius[i];
        double dvx = dtm*f[i][0];
        double dvy = dtm*f[i][1];
        double dvz = dtm*torque[i][2]/(0.5*radi*radi);

        // add the drag, tilt, and forcing
        double vpx = v[i][0] + dvx + dtm*(bvector[i][0] + forcing[i][0]);
        double vpy = v[i][1] + dvy + dtm*(bvector[i][1] + forcing[i][1]);

        // add acceleration due to ocean drag
        double ovx = ocean_vel[i][0];
        double ovy = ocean_vel[i][1];
        double wx = ovx - vpx;
        double wy = ovy - vpy;

        // get the exchange matrix A = {{1+a,-a},{-b,1+b}}
        double ice_volume = ice_area[i]*mean_thickness[i];
        double ice_density = rmass[i]/ice_volume;
        double D12 = ocean_density*ocean_drag*sqrt(wx*wx + wy*wy)*ice_area[i];
        double a = D12*dtf/(ice_density*ice_volume);

        // one-way coupling
        vpx = (vpx + a*ovx)/(1. + a); // one-way coupling
        vpy = (vpy + a*ovy)/(1. + a);

        // two-way coupling
//      double b = D12*dtf/(ocean_density*ice_volume);
//      vpx = ((1. + b)*vpx + a*ovx)/(1. + a + b);
//      vpy = ((1. + b)*vpy + a*ovy)/(1. + a + b);

        // half step coriolis acceleration
        double cdt = coriolis[i]*dtf;
        double den = 1. + cdt*cdt;
            v[i][0] = (vpx + cdt*vpy)/den;
            v[i][1] = (vpy - cdt*vpx)/den;
        omega[i][2] = omega[i][2] + dvz;

        x[i][0] += dtv * v[i][0];  // full step with explicit vel
        x[i][1] += dtv * v[i][1];

      } // end if (mask[i] & groupbit)

      if (!(mask[i]&groupbit)){
            vn[i][0] =     vn[i][1] =     vn[i][2] = 0.;
        torque[i][0] = torque[i][1] = torque[i][2] = 0.;
         omega[i][0] =  omega[i][1] =  omega[i][2] = 0.;
      }
 
    } // end for (int i = 0; i < nlocal; i++)

} // end void FixNVESphereDemsi::initial_integrate(int /*vflag*/)
/* ---------------------------------------------------------------------- */

void FixNVESphereDemsi::final_integrate()
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  double *ice_area = atom->ice_area;
  double *coriolis = atom->coriolis;
  double **ocean_vel = atom->ocean_vel;
  double **bvector = atom->bvector;
  double **forcing = atom->forcing;
  double **vn = atom->vn; // adding vn
  double *mean_thickness = atom->mean_thickness;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // set timestep here since dt may have changed or come via rRESPA
  double dtv = update->dt;
  double dtf = 0.5*dtv;

    double rke = 0.0;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {

            v[i][0] = vn[i][0];
            v[i][1] = vn[i][1];
        omega[i][2] = vn[i][2];

        // half step acceleration with time(n) forces
        double dtm = dtf/rmass[i];
        double radi = radius[i];
        double dvx = dtm*f[i][0];
        double dvy = dtm*f[i][1];
        double dvz = dtm*torque[i][2]/(0.5*radi*radi);

        // add the wind, tilt, and forcing
        double vpx = v[i][0] + dvx + dtm*(bvector[i][0] + forcing[i][0]);
        double vpy = v[i][1] + dvy + dtm*(bvector[i][1] + forcing[i][1]);

        // add acceleration due to ocean drag
        double ovx = ocean_vel[i][0];
        double ovy = ocean_vel[i][1];
        double wx = ovx - vpx;
        double wy = ovy - vpy;

        // get the exchange matrix A = {{1+a,-a},{-b,1+b}}
        double ice_volume = ice_area[i]*mean_thickness[i];
        double ice_density = rmass[i]/ice_volume;
        double D12 = ocean_density*ocean_drag*sqrt(wx*wx + wy*wy)*ice_area[i];
        double a = D12*dtf/(ice_density*ice_volume);

        // one-way coupling
        vpx = (vpx + a*ovx)/(1. + a); // one-way coupling
        vpy = (vpy + a*ovy)/(1. + a);

        // two-way coupling
//      double b = D12*dtf/(ocean_density*ice_volume);
//      vpx = ((1. + b)*vpx + a*ovx)/(1. + a + b);
//      vpy = ((1. + b)*vpy + a*ovy)/(1. + a + b);

        // half step coriolis acceleration
        double cdt = coriolis[i]*dtf;
        double den = 1. + cdt*cdt;
            v[i][0] = (vpx + cdt*vpy)/den;
            v[i][1] = (vpy - cdt*vpx)/den;
        omega[i][2] = omega[i][2] + dvz;

//      x[i][0] += dtf * v[i][0];  // half step with time(n+1) vel
//      x[i][1] += dtf * v[i][1];

        rke += (omega[i][2]*omega[i][2])*radi*radi*rmass[i];

      } // end if (mask[i] & groupbit)

      if (!(mask[i]&groupbit)){
            vn[i][0] =     vn[i][1] =     vn[i][2] = 0.;
        torque[i][0] = torque[i][1] = torque[i][2] = 0.;
         omega[i][0] =  omega[i][1] =  omega[i][2] = 0.;
      }
 
    } // end for (int i = 0; i < nlocal; i++)

    for (int i = 0; i < atom->nlocal; i++) {
        vn[i][0] =  omega[i][0]; // integral dLogV
        vn[i][1] =  omega[i][1]; // integral dLogS
        vn[i][2] = torque[i][0]; // dMach
    } // end for (int i = 0; i < atom->nlocal; i++)

} // end void FixNVESphereDemsi::final_integrate()

/* ---------------------------------------------------------------------- */

