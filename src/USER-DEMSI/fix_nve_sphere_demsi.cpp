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
  double vnx, vny, vnz, dvx, dvy, dvz, vpx, vpy;

  if (timeIntegrationFlag == 0) { // Explicit Verlet form

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {

        // full step displacement with time(n) velocity
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];

        // half step acceleration with time(n) forces
        double dtm = dtf/rmass[i];
        double radi = radius[i];
        dvx = dtm*f[i][0];
        dvy = dtm*f[i][1];
        dvz = dtm*torque[i][2]/(0.5*radi*radi);

        vpx = v[i][0] + dvx;
        vpy = v[i][1] + dvy;
        double wx = ocean_vel[i][0] - vpx;
        double wy = ocean_vel[i][1] - vpy;
        
        vel_diff = sqrt(wx*wx + wy*wy);
        D = ice_area[i]*ocean_drag*ocean_density*vel_diff;
        m_prime = rmass[i]/dtf;
        a00 = a11 = m_prime+D;
        a10 = rmass[i]*coriolis[i];
        a01 = -a10;

        b0 = m_prime*vpx + bvector[i][0] + forcing[i][0] + D*ocean_vel[i][0];
        b1 = m_prime*vpy + bvector[i][1] + forcing[i][1] + D*ocean_vel[i][1];

        detinv = 1.0/(a00*a11 - a01*a10);
        v[i][0] = detinv*( a11*b0 - a01*b1);
        v[i][1] = detinv*(-a10*b0 + a00*b1);

            v[i][0] = detinv*( a11*b0 - a01*b1);
            v[i][1] = detinv*(-a10*b0 + a00*b1);
        omega[i][2] += dvz;

        vn[i][0] =     v[i][0]; // for debugging
        vn[i][1] =     v[i][1];
        vn[i][2] = omega[i][2];

      } // end if (mask[i] & groupbit)

      if (!(mask[i]&groupbit)){
           vn[i][0] =    vn[i][1] =    vn[i][2] = 0.;
            v[i][0] =     v[i][1] =     v[i][2] = 0.;
        omega[i][0] = omega[i][1] = omega[i][2] = 0.;
      }
 
    } // end for (int i = 0; i < nlocal; i++)

  } else if (timeIntegrationFlag == 1) { // Explicit Rate form

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {

        vn[i][0] =     v[i][0];
        vn[i][1] =     v[i][1];
        vn[i][2] = omega[i][2];

        // half step displacement with time(n) velocity
        x[i][0] += dtf * v[i][0];
        x[i][1] += dtf * v[i][1];

        // full step acceleration with time(n) forces
        double dtm = dtv/rmass[i];
        double radi = radius[i];
        dvx = dtm*f[i][0];
        dvy = dtm*f[i][1];
        dvz = dtm*torque[i][2]/(0.5*radi*radi);

        vpx = v[i][0] + dvx;
        vpy = v[i][1] + dvy;
        double wx = ocean_vel[i][0] - vpx;
        double wy = ocean_vel[i][1] - vpy;
        
        vel_diff = sqrt(wx*wx + wy*wy);
        D = ice_area[i]*ocean_drag*ocean_density*vel_diff;
        m_prime = rmass[i]/dtv;
        a00 = a11 = m_prime+D;
        a10 = rmass[i]*coriolis[i];
        a01 = -a10;

        b0 = m_prime*vpx + bvector[i][0] + forcing[i][0] + D*ocean_vel[i][0];
        b1 = m_prime*vpy + bvector[i][1] + forcing[i][1] + D*ocean_vel[i][1];

        detinv = 1.0/(a00*a11 - a01*a10);
        v[i][0] = detinv*( a11*b0 - a01*b1);
        v[i][1] = detinv*(-a10*b0 + a00*b1);

            v[i][0] = detinv*( a11*b0 - a01*b1);
            v[i][1] = detinv*(-a10*b0 + a00*b1);
        omega[i][2] += dvz;

      } // end if (mask[i] & groupbit)

      if (!(mask[i]&groupbit)){
           vn[i][0] =    vn[i][1] =    vn[i][2] = 0.;
            v[i][0] =     v[i][1] =     v[i][2] = 0.;
        omega[i][0] = omega[i][1] = omega[i][2] = 0.;
      }
 
    } // end for (int i = 0; i < nlocal; i++)
    
  } else if (timeIntegrationFlag == 2) { // Implicit Rate form

    double vnmag, dvmag, dvfac;
    double elastic_wavespeed = sqrt(bulkModulus/rho0) + sqrt(shearModulus/rho0);
//  double bulk_CFL = fmax(1., bulkCFL);
    double rcfac = 500.;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {

        // full step displacement
        double dx = dtv*v[i][0];
        double dy = dtv*v[i][1];

        // full step acceleration with time(n) forces
        double dtm = dtv/rmass[i];
        double radi = radius[i];
        dvx = dtm*f[i][0];
        dvy = dtm*f[i][1];
        dvz = dtm*torque[i][2]/(0.5*radi*radi);

        // limit accelerations and displacements
        dvmag = dvx*dvx + dvy*dvy + (radi*dvz)*(radi*dvz);
        double cfc = sqrt(dvmag)*(dtv/radi);  // convective cfl
        double bulk_CFL = fmax(1., elastic_wavespeed*dtv/radi);
        dvfac = fmax(1., fmax(1., rcfac/(bulk_CFL*bulk_CFL))*cfc);
        if(bulk_CFL >= 1.) {dvx /= dvfac; dvy /= dvfac; dvz /= dvfac; dx /= dvfac; dy /= dvfac;}
        
        x[i][0] += dx;
        x[i][1] += dy;

        vpx = v[i][0] + dvx;
        vpy = v[i][1] + dvy;
        double wx = ocean_vel[i][0] - vpx;
        double wy = ocean_vel[i][1] - vpy;

        vel_diff = sqrt(wx*wx + wy*wy);
        D = ice_area[i]*ocean_drag*ocean_density*vel_diff;
        m_prime = rmass[i]/dtv;
        a00 = a11 = m_prime+D;
        a10 = rmass[i]*coriolis[i];
        a01 = -a10;

        b0 = m_prime*vpx + bvector[i][0] + forcing[i][0] + D*ocean_vel[i][0];
        b1 = m_prime*vpy + bvector[i][1] + forcing[i][1] + D*ocean_vel[i][1];

        detinv = 1.0/(a00*a11 - a01*a10);
        v[i][0] = detinv*( a11*b0 - a01*b1);
        v[i][1] = detinv*(-a10*b0 + a00*b1);

            v[i][0] = detinv*( a11*b0 - a01*b1);
            v[i][1] = detinv*(-a10*b0 + a00*b1);
        omega[i][2] += dvz;

      } // end if (mask[i] & groupbit)

      if (!(mask[i]&groupbit)){
           vn[i][0] =    vn[i][1] =    vn[i][2] = 0.;
            v[i][0] =     v[i][1] =     v[i][2] = 0.;
        omega[i][0] = omega[i][1] = omega[i][2] = 0.;
      }
 
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
  double vnx, vny, vnz, dvx, dvy, dvz, vpx, vpy;

  if (timeIntegrationFlag == 0) { // Explicit Verlet form
  
    double rke = 0.0;
    for (int i = 0; i < nlocal; i++){
      if (mask[i] & groupbit) {

//      x[i][0] += dtf * v[i][0];
//      x[i][1] += dtf * v[i][1];

        // half step acceleration
        double dtm = dtf/rmass[i];
        double radi = radius[i];
        dvx = dtm*f[i][0];
        dvy = dtm*f[i][1];
        dvz = dtm*torque[i][2]/(0.5*radi*radi);

        vpx = v[i][0] + dvx;
        vpy = v[i][1] + dvy;
        double wx = ocean_vel[i][0] - vpx;
        double wy = ocean_vel[i][1] - vpy;
        
        vel_diff = sqrt(wx*wx + wy*wy);
        D = ice_area[i]*ocean_drag*ocean_density*vel_diff;
        m_prime = rmass[i]/dtf; // half step with time(n+1) forces
        a00 = a11 = m_prime+D;
        a10 = rmass[i]*coriolis[i];
        a01 = -a10;

        b0 = m_prime*vpx + bvector[i][0] + forcing[i][0] + D*ocean_vel[i][0];
        b1 = m_prime*vpy + bvector[i][1] + forcing[i][1] + D*ocean_vel[i][1];

        detinv = 1.0/(a00*a11 - a01*a10);
        v[i][0] = detinv*( a11*b0 - a01*b1);
        v[i][1] = detinv*(-a10*b0 + a00*b1);

            v[i][0] = detinv*( a11*b0 - a01*b1);
            v[i][1] = detinv*(-a10*b0 + a00*b1);
        omega[i][2] += dvz;

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
//      omega[i][2] = vn[i][2];

        x[i][0] += dtf * v[i][0];
        x[i][1] += dtf * v[i][1];

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
//      omega[i][2] = vn[i][2];

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

