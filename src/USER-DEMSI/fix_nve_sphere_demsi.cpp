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

  ocean_density = ocean_drag = 0;

  timeIntegrationFlag = 0;
  bulkCFL = 1.0;

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

  double D, vel_diff, m_prime;
  double a00, a01, a10, a11;
  double detinv;
  double b0, b1;
  double det;

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // set timestep here since dt may have changed or come via rRESPA

  double dtv = update->dt;
  double dtf = 0.5 * update->dt;
  const double dvzero = 1.e-30;
  double vnmag, dvmag, dvfac, vnx, vny, vnz, dvx, dvy, dvz, vpx, vpy;

  if (timeIntegrationFlag == 0) { // Explicit Verlet form

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {

        x[i][0] += dtv * v[i][0]; // half step with time(n) velocity
        x[i][1] += dtv * v[i][1];

        vel_diff = sqrt((ocean_vel[i][0]-v[i][0])*(ocean_vel[i][0]-v[i][0]) +
          (ocean_vel[i][1]-v[i][1])*(ocean_vel[i][1]-v[i][1]));
        D = ice_area[i]*ocean_drag*ocean_density*vel_diff;
        m_prime = rmass[i]/dtf; // half step with time(n) forces
        a00 = a11 = m_prime+D;
        a10 = rmass[i]*coriolis[i];
        a01 = -a10;

        b0 = m_prime*v[i][0] + f[i][0] + bvector[i][0] + forcing[i][0] + D*ocean_vel[i][0];
        b1 = m_prime*v[i][1] + f[i][1] + bvector[i][1] + forcing[i][1] + D*ocean_vel[i][1];

        detinv = 1.0/(a00*a11 - a01*a10);
        v[i][0] = detinv*( a11*b0 - a01*b1);
        v[i][1] = detinv*(-a10*b0 + a00*b1);

        double dtm = dtf/rmass[i];
        double radi = radius[i];
        omega[i][2] += dtm* torque[i][2]/(0.5*radi*radi);

      } // end if (mask[i] & groupbit)
    } // end for (int i = 0; i < nlocal; i++)

  } else if (timeIntegrationFlag == 1) { // Explicit Rate form

    // save time(n) v into vn for use in PairGranRate::compute
    for (int i = 0; i < atom->nlocal; i++) {
        vn[i][0] =     v[i][0];
        vn[i][1] =     v[i][1];
        vn[i][2] = omega[i][2];
    } // end for (int i = 0; i < atom->nlocal; i++

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {

        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];

        // acceleration due to forces and torques
        double dtm = dtv/rmass[i];
        double radi = radius[i];
        dvx = dtm*f[i][0];
        dvy = dtm*f[i][1];
        dvz = dtm*torque[i][2]/(0.5*radi*radi);

        vnx = vn[i][0];
        vny = vn[i][1];
        vnz = vn[i][2];

        vpx = vnx + dvx;
        vpy = vny + dvy;

        // add acceleration due to ocean drag
        double wx = ocean_vel[i][0] - vpx;
        double wy = ocean_vel[i][1] - vpy;
        
        // get the convective CFL based on relative velocity
        double convCFL = ocean_drag*dtm*sqrt(wx*wx + wy*wy)
                       *(ocean_density*MY_PI*radi*radi*ice_area[i]);
//      double convCFL = ocean_drag*dtm*sqrt(wx*wx + wy*wy)
//                     *(ocean_density*ice_area[i]);

        wx = 0.5*wx + vpx; // vx in the limit dt->oo
        wy = 0.5*wy + vpy; // vy in the limit dt->oo

        double cfx = exp(-convCFL);
        vpx = wx + (vpx - wx)*cfx;
        vpy = wy + (vpy - wy)*cfx;

        // add the other accelerations
        m_prime = 1.0/dtm;
        a00 = a11 = m_prime;
        a10 = rmass[i]*coriolis[i];
        a01 = -a10;
        b0 = m_prime*vpx + bvector[i][0] + forcing[i][0];
        b1 = m_prime*vpy + bvector[i][1] + forcing[i][1];
        detinv = 1.0/(a00*a11 - a01*a10);

            v[i][0] = detinv*( a11*b0 - a01*b1);
            v[i][1] = detinv*(-a10*b0 + a00*b1);
        omega[i][2] = vnz + dvz;

      } // end if (mask[i] & groupbit)
    } // end for (int i = 0; i < nlocal; i++)
    
  } else if (timeIntegrationFlag == 2) { // Implicit Rate form

    for (int i = 0; i < atom->nlocal; i++) {
        vn[i][0] =     v[i][0];
        vn[i][1] =     v[i][1];
        vn[i][2] = omega[i][2];
    } // end for (int i = 0; i < atom->nlocal; i++

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {

        // limit the TOTAL displacement to a local strain
        double CFLfac = 2000.;
        double dx = dtv*v[i][0];
        double dy = dtv*v[i][1];
        double dxfac = CFLfac*sqrt(dx*dx + dy*dy)/radius[i];
 
        // acceleration due to forces and torques
        double dtm = dtv/rmass[i];
        double radi = radius[i];
        dvx = dtm*f[i][0];
        dvy = dtm*f[i][1];
        dvz = dtm*torque[i][2]/(0.5*radi*radi);

        vnx = vn[i][0];
        vny = vn[i][1];
        vnz = vn[i][2];

        // limit accelerations due to forces and torques, and displacements
        vnmag = vnx*vnx + vny*vny + (radi*vnz)*(radi*vnz) + dvzero;
        dvmag = dvx*dvx + dvy*dvy + (radi*dvz)*(radi*dvz);
        dvfac = 2.*sqrt(dvmag/vnmag);

        if (dvfac > 1.0) {
          dvx /= dvfac; dvy /= dvfac; dvz /= dvfac; dx /= dvfac; dy /= dvfac;
        }
        
        x[i][0] += dx;
        x[i][1] += dy;

        vpx = vnx + dvx;
        vpy = vny + dvy;

        // add acceleration due to ocean drag
        double wx = ocean_vel[i][0] - vpx;
        double wy = ocean_vel[i][1] - vpy;
        
        // get the convective CFL based on relative velocity
        double convCFL = ocean_drag*dtm*sqrt(wx*wx + wy*wy)
                       *(ocean_density*MY_PI*radi*radi*ice_area[i]);
//      double convCFL = ocean_drag*dtm*sqrt(wx*wx + wy*wy)
//                     *(ocean_density*ice_area[i]);

        wx = 0.5*wx + vpx; // vx in the limit dt->oo
        wy = 0.5*wy + vpy; // vy in the limit dt->oo

        double cfx = exp(-convCFL);
        vpx = wx + (vpx - wx)*cfx;
        vpy = wy + (vpy - wy)*cfx;

        // add the other accelerations
        m_prime = 1.0/dtm;
        a00 = a11 = m_prime;
        a10 = rmass[i]*coriolis[i];
        a01 = -a10;
        b0 = m_prime*vpx + bvector[i][0] + forcing[i][0];
        b1 = m_prime*vpy + bvector[i][1] + forcing[i][1];
        detinv = 1.0/(a00*a11 - a01*a10);

            v[i][0] = detinv*( a11*b0 - a01*b1);
            v[i][1] = detinv*(-a10*b0 + a00*b1);
        omega[i][2] = vnz + dvz;

      } // end if (mask[i] & groupbit)
    } // end for (int i = 0; i < nlocal; i++)
  } else {
    error->all(FLERR,"int(timeIntegrationFlag) must be (0,1,2)");
  }
} // end void FixNVESphereDemsi::initial_integrate(int /*vflag*/)
/* ---------------------------------------------------------------------- */

void FixNVESphereDemsi::final_integrate()
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *rmass = atom->rmass;
  double *ice_area = atom->ice_area;
  double *coriolis = atom->coriolis;
  double **ocean_vel = atom->ocean_vel;
  double **bvector = atom->bvector;
  double **forcing = atom->forcing;
  double **vn = atom->vn; // adding vn

  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double D, vel_diff, m_prime;
  double a00, a01, a10, a11;
  double detinv;
  double b0, b1;
  double det;

  // set timestep here since dt may have changed or come via rRESPA

  double dtv = update->dt;
  double dtf = 0.5 * update->dt;

  if (timeIntegrationFlag == 0) { // Explicit Verlet form
  
    double rke = 0.0;
    for (int i = 0; i < nlocal; i++){
      if (mask[i] & groupbit) {
        vel_diff = sqrt((ocean_vel[i][0]-v[i][0])*(ocean_vel[i][0]-v[i][0]) +
            (ocean_vel[i][1]-v[i][1])*(ocean_vel[i][1]-v[i][1]));
        D = ice_area[i]*ocean_drag*ocean_density*vel_diff;
        m_prime = rmass[i]/dtf; // half step with time(n+1) forces
        a00 = a11 = m_prime+D;
        a10 = rmass[i]*coriolis[i];
        a01 = -a10;

        b0 = m_prime*v[i][0] + f[i][0] + bvector[i][0] + forcing[i][0] + D*ocean_vel[i][0];
        b1 = m_prime*v[i][1] + f[i][1] + bvector[i][1] + forcing[i][1] + D*ocean_vel[i][1];

        detinv = 1.0/(a00*a11 - a01*a10);
        v[i][0] = detinv*( a11*b0 - a01*b1);
        v[i][1] = detinv*(-a10*b0 + a00*b1);

        double dtm = dtf/rmass[i];
        double radi = radius[i];
        omega[i][2] += dtm* torque[i][2]/(0.5*radi*radi);

        rke += (omega[i][0]*omega[i][0] + omega[i][1]*omega[i][1] +
                omega[i][2]*omega[i][2])*radius[i]*radius[i]*rmass[i];
      } // end if (mask[i] & groupbit)
    } // end for (int i = 0; i < nlocal; i++) 

    for (int i = 0; i < atom->nlocal; i++) {
        vn[i][0] = omega[i][0]; // integral number of compressive yields
        vn[i][1] = omega[i][1]; // integral number of       shear yields
        vn[i][2] =     v[i][2]; // integral number of     tensile fractures
    } // end for (int i = 0; i < atom->nlocal; i++)

  } else if (timeIntegrationFlag == 1) { // Explicit Rate form

    double rke = 0.0;
    for (int i = 0; i < nlocal; i++){
      if (mask[i] & groupbit) {

            v[i][0] = vn[i][0];
            v[i][1] = vn[i][1];
        omega[i][2] = vn[i][2];

        rke += (omega[i][0]*omega[i][0] + omega[i][1]*omega[i][1] +
                omega[i][2]*omega[i][2])*radius[i]*radius[i]*rmass[i];
      } // end  if (mask[i] & groupbit)
    } // end for (int i = 0; i < nlocal; i++)

    for (int i = 0; i < atom->nlocal; i++) {
        vn[i][0] = omega[i][0]; // integral number of compressive yields
        vn[i][1] = omega[i][1]; // integral number of       shear yields
        vn[i][2] =     v[i][2]; // integral number of     tensile fractures
    } // end for (int i = 0; i < atom->nlocal; i++)

  } else if (timeIntegrationFlag == 2) { // Implicit Rate form

    double rke = 0.0;
    for (int i = 0; i < nlocal; i++){
      if (mask[i] & groupbit) {

            v[i][0] = vn[i][0];
            v[i][1] = vn[i][1];
        omega[i][2] = vn[i][2];

        rke += (omega[i][0]*omega[i][0] + omega[i][1]*omega[i][1] +
                omega[i][2]*omega[i][2])*radius[i]*radius[i]*rmass[i];

      } // end  if (mask[i] & groupbit)
    } // end for (int i = 0; i < nlocal; i++)

    for (int i = 0; i < atom->nlocal; i++) {
        vn[i][0] = omega[i][0]; // integral number of compressive yields
        vn[i][1] = omega[i][1]; // integral number of       shear yields
        vn[i][2] =     v[i][2]; // integral number of     tensile fractures
    } // end for (int i = 0; i < atom->nlocal; i++)

  } else {
    error->all(FLERR,"int(timeIntegrationFlag) must be (0,1,2)");
  }
} // end void FixNVESphereDemsi::final_integrate()

/* ---------------------------------------------------------------------- */

