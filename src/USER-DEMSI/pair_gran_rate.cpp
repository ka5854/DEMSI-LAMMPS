/* ----------------------------------------------------------------------
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   ------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
   Contributing authors: Dan S. Bolintineanu (SNL), Adrian K. Turner (LANL),
   B. A. Kashiwa (LASL)
   ------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "pair_gran_rate.h"
#include "atom.h"
#include "atom_vec.h"
#include "update.h"
#include "force.h"
#include "fix.h"
#include "fix_neigh_history.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"

#include "group.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define DEBUGID_1 35
#define DEBUGID_2 55
#define DEBUG_TIMESTEP 32696
/* ---------------------------------------------------------------------- */

PairGranRate::PairGranRate(LAMMPS *lmp) :
              PairGranHookeHistory(lmp, 12)
{
  nondefault_history_transfer = 1;
  beyond_contact = 1;
}

/* ---------------------------------------------------------------------- */
PairGranRate::~PairGranRate()
{
}
/* ---------------------------------------------------------------------- */

void PairGranRate::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  // update rigid body info for owned & ghost atoms if using FixRigid masses
  // body[i] = which body atom I is in, -1 if none
  // mass_body = mass of each rigid body
  // Not yet applicable for DEMSI, but may be something to look into

  if (fix_rigid && neighbor->ago == 0){
    int i;
    int tmp;
    int *body = (int *) fix_rigid->extract("body",tmp);
    double *mass_body = (double *) fix_rigid->extract("masstotal",tmp);
    if (atom->nmax > nmax) {
      memory->destroy(mass_rigid);
      nmax = atom->nmax;
      memory->create(mass_rigid,nmax,"pair:mass_rigid");
    }
    int nlocal = atom->nlocal;
    for (i = 0; i < nlocal; i++)
      if (body[i] >= 0) mass_rigid[i] = mass_body[body[i]];
      else mass_rigid[i] = 0.0;
    comm->forward_comm_pair(this);
  }

//if (int(timeIntegrationFlag) == 1) {
    compute_rate_explicit();
//} else {
//  compute_rate_implicit();
//}

  if (vflag_fdotr) virial_fdotr_compute();

} // end PairGranRate::compute

void PairGranRate::compute_rate_explicit()
{
  int i,j,ii,jj,inum,jnum;
  int itype,jtype;

  int *ilist,*jlist,*numneigh,**firstneigh;
  int **firsttouch;
  double *history,*allhistory,**firsthistory;
  
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  firsttouch = fix_history->firstflag;
  firsthistory = fix_history->firstvalue;
  
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **vn = atom->vn;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

//if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  int bondFlagIn; // incoming bondFlag
  int Fplus;      // outgoing bondFlag

  double delx, dely, rsq, radsum;
  double r, rinv, nx, ny, Rstar;
  double vrn, vrt;
  double hAVGi, hAVGj, hDeltal;
  double fx, fy, fz;
  double Pplus, Splus; // outgoing bondPressure, bondShear

  bool touchflag = false;
//const bool historyupdate = (update->setupflag) ? false : true;

  int historyupdate = 1;
  if (update->setupflag) historyupdate = 0;

  double dtv = update->dt;
  double dtf = 0.5 * update->dt; // * force->ftm2v;

  /* Load the explicit force/torque */
  
    for (int i = 0; i < nlocal; i++){
      f[i][0] = 0.0;
      f[i][1] = 0.0;
//    f[i][2] = 0.0;
//    torque[i][0] = 0.0;
//    torque[i][1] = 0.0;
      torque[i][2] = 0.0;
    }
  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    allhistory = firsthistory[i];

    // loop over neighbors of each element
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      jtype = atom->type[j];
      j &= NEIGHMASK;

//    *touch = &firsttouch[i][jj];
      history = &allhistory[size_history*jj];

      /*'history' now points to the ii-jj array that stores
         all the history associated with pair ii-jj
         For all pairs:
         *history[0]: bondPressure
         *history[1]: bondShear
         *history[2]: bondFlag [0=no bond, 1=bonded]]
      */

//    delx = x[i][0] - x[j][0]; // hopkins uses inward normal relative to [i]
//    dely = x[i][1] - x[j][1];
      delx = x[j][0] - x[i][0]; // rateform uses outward normal relative to [i] ??
      dely = x[j][1] - x[i][1];
      rsq = delx*delx + dely*dely;
//    radsum = LapOver*(radius[i] + radius[j]);
      radsum = 1.001*(radius[i] + radius[j]);

      bondFlagIn = int(history[2]);
      Fplus = bondFlagIn;

      if(!bondFlagIn){
        if (rsq >= radsum*radsum){ // no contact
          firsttouch[i][jj] = 0;
          Fplus = 0;
          fx = fy = fz = 0.;
        } else { // in first contact
          firsttouch[i][jj] = 1;
          Fplus = 1;
        } // end if(rsq >= radsum*radsum
        history[0] = history[1] = 0.;
      } // end if(!bondFlagIn)

//    if(Fplus){
        r = sqrt(rsq);
        rinv = 1.0/r;
        nx = delx*rinv;
        ny = dely*rinv;
        Rstar = 2*(radius[i] * radius[j])
                 /(radius[i] + radius[j]);

//      vrn = (vn[j][0] - vn[i][0])*nx + (vn[j][1] - vn[i][1])*ny;
//      vrt = (vn[j][2] + vn[i][2])*Rstar;
        vrn = (v[j][0] - v[i][0])*nx + (v[j][1] - v[i][1])*ny;
        vrt = (omega[j][2] + omega[i][2])*Rstar;

        if(bondFlagIn or Fplus){
          Pplus = history[0]  - bulkModulus*dtv*vrn*rinv;
          Splus = history[1] - shearModulus*dtv*vrt*rinv*2.;
        } else {
          Pplus = 0.;
          Splus = 0.;
        }

        hAVGi = rmass[i]/(rho0*MY_PI*radius[i]*radius[i]);
        hAVGj = rmass[j]/(rho0*MY_PI*radius[j]*radius[j]);

        hDeltal = 2*MY_PI3*(hAVGi*radius[i] * hAVGj*radius[j])
                          /(hAVGi*radius[i] + hAVGj*radius[j]);

        fx = Pplus*nx*hDeltal;
        fy = Pplus*ny*hDeltal;
        fz = Splus   *hDeltal*Rstar;

        f[i][0] -= fx;
        f[i][1] -= fy;
        torque[i][2] -= fz;

        if (force->newton_pair || j < atom->nlocal){
          f[j][0] += fx;
          f[j][1] += fy;
          torque[j][2] += fz;
        }
        
        if (int(timeIntegrationFlag) == 0){
        
        // test for tensile fracture, compressive flowstress, shear flowstress

        if (Pplus <= tensileFractureStress) { // tensile fracture
          Pplus = 0.;
          Fplus = 0;
        } else if (Pplus > compressiveYieldStress) { // perfectly plastic compressive flowstress
          Pplus = compressiveYieldStress;
        }
        if (signbit(Splus)) { // Splus negative
          Splus = fmax(Splus, -shearYieldStress);
        } else { // Splus positive
          Splus = fmin(Splus, shearYieldStress);
        }

      // update history
        if(historyupdate){
          history[0] = Pplus;
          history[1] = Splus;
          history[2] = double(Fplus);
        }
        }
//    } // if(Fplus)

    } // end for (ii = 0; ii < inum; ii++)
  } // end for (ii = 0; ii < inum; ii++)

if (int(timeIntegrationFlag) == 0) return;

  /*! Put the new velocity into vn for fix_nve_sphere_demsi::integrate_final.
      Recall that fix_nve_sphere_demsi:integrate_initial has put the explicit
      part of the update into {v[0,1],omega[2]}. */
  for (int i = 0; i < nlocal; i++) {
// if (mask[i] & groupbit) {

      vn[i][0] =     v[i][0] + dtf*     f[i][0]/rmass[i];
      vn[i][1] =     v[i][1] + dtf*     f[i][1]/rmass[i];
      vn[i][2] = omega[i][2] + dtf*torque[i][2]/(0.5*rmass[i]*radius[i]*radius[i]);

//  }
  } // end for (int i = 0; i < nlocal; i++)

  /* Load the explicit force/torque */
  
    for (int i = 0; i < nlocal; i++){
      f[i][0] = 0.0;
      f[i][1] = 0.0;
//    f[i][2] = 0.0;
//    torque[i][0] = 0.0;
//    torque[i][1] = 0.0;
      torque[i][2] = 0.0;
    }

  /* Back-substitute for the stress increments based on vn; update the
     history stress; and reload the forces/torques for use in the next call
     to fix_nve_sphere_demsi::integrate_initial */

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    allhistory = firsthistory[i];

    // loop over neighbors of each element
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      jtype = atom->type[j];
      j &= NEIGHMASK;

//    *touch = &firsttouch[i][jj];
      history = &allhistory[size_history*jj];

      /*'history' now points to the ii-jj array that stores
         all the history associated with pair ii-jj
         For all pairs:
         *history[0]: bondPressure
         *history[1]: bondShear
         *history[2]: bondFlag [0=no bond, 1=bonded]]
      */
      
//    delx = x[i][0] - x[j][0]; // hopkins uses inward normal relative to [i]
//    dely = x[i][1] - x[j][1];
      delx = x[j][0] - x[i][0]; // rateform uses outward normal relative to [i] ??
      dely = x[j][1] - x[i][1];
      rsq = delx*delx + dely*dely;
//    radsum = LapOver*(radius[i] + radius[j]);
      radsum = 1.001*(radius[i] + radius[j]);

      bondFlagIn = int(history[2]);
      Fplus = bondFlagIn;

      if(!bondFlagIn){
        if (rsq >= radsum*radsum){ // no contact
          firsttouch[i][jj] = 0;
          Fplus = 0;
          fx = fy = fz = 0.;
        } else { // in first contact
          firsttouch[i][jj] = 1;
          Fplus = 1;
        } // end if(rsq >= radsum*radsum
        history[0] = history[1] = 0.;
      } // end if(!bondFlagIn)

//    if(Fplus){
        r = sqrt(rsq);
        rinv = 1.0/r;
        nx = delx*rinv;
        ny = dely*rinv;
        Rstar = 2*(radius[i] * radius[j])
                 /(radius[i] + radius[j]);

        vrn = (vn[j][0] - vn[i][0])*nx + (vn[j][1] - vn[i][1])*ny;
        vrt = (vn[j][2] + vn[i][2])*Rstar;

        if(bondFlagIn or Fplus){
          Pplus = history[0]  - bulkModulus*dtv*vrn*rinv;
          Splus = history[1] - shearModulus*dtv*vrt*rinv*2.;
        } else {
          Pplus = 0.;
          Splus = 0.;
        }
    
      /*! Reload {f,torque} for the next integrate_initial */

        hAVGi = rmass[i]/(rho0*MY_PI*radius[i]*radius[i]);
        hAVGj = rmass[j]/(rho0*MY_PI*radius[j]*radius[j]);

        hDeltal = 2*MY_PI3*(hAVGi*radius[i] * hAVGj*radius[j])
                          /(hAVGi*radius[i] + hAVGj*radius[j]);

        fx = Pplus*nx*hDeltal;
        fy = Pplus*ny*hDeltal;
        fz = Splus   *hDeltal*Rstar;
        
//    } // end if(Fplus)
      
        f[i][0] -= fx;
        f[i][1] -= fy;
        torque[i][2] -= fz;

        if (force->newton_pair || j < atom->nlocal){
          f[j][0] += fx;
          f[j][1] += fy;
          torque[j][2] += fz;
        }

        // test for tensile fracture, compressive flowstress, shear flowstress

        if (Pplus <= tensileFractureStress) { // tensile fracture
          Pplus = 0.;
          Fplus = 0;
        } else if (Pplus > compressiveYieldStress) { // perfectly plastic compressive flowstress
          Pplus = compressiveYieldStress;
        }
        if (signbit(Splus)) { // Splus negative
          Splus = fmax(Splus, -shearYieldStress);
        } else { // Splus positive
          Splus = fmin(Splus, shearYieldStress);
        }

      // update history
        if(historyupdate){
          history[0] = Pplus;
          history[1] = Splus;
          history[2] = double(Fplus);
        }

      if (evflag) ev_tally_xyz(i,j,atom->nlocal, force->newton_pair,
              0.0,0.0,fx,fy,0,x[i][0]-x[j][0],x[i][1]-x[j][1],0);

    } // end for (ii = 0; ii < inum; ii++)
  } // end for (ii = 0; ii < inum; ii++)
} // end PairGranRate::compute_rate_explicit()

/* ----------------------------------------------------------------------
   global settings
   ------------------------------------------------------------------------- */

void PairGranRate::settings(int narg, char **arg)
{
  if (narg != 7) error->all(FLERR,"Illegal pair_style command");
             bulkModulus = force->numeric(FLERR, arg[0]);
            shearModulus = force->numeric(FLERR, arg[1]);
        shearYieldStress = force->numeric(FLERR, arg[2]);
  compressiveYieldStress = force->numeric(FLERR, arg[3]);
   tensileFractureStress = force->numeric(FLERR, arg[4]);
     timeIntegrationFlag = force->numeric(FLERR, arg[5]);
                 bulkCFL = force->numeric(FLERR, arg[6]);
}

/* ---------------------------------------------------------------------- */
double PairGranRate::init_one(int i, int j)
{
  double cutoff;
  cutoff = PairGranHookeHistory::init_one(i, j);
  cutoff += maxrad_dynamic[i]*0.1; //This could be an input parameter?
  return cutoff;
}

/* ---------------------------------------------------------------------- */

double PairGranRate::single(int i, int j, int itype, int jtype,
    double rsq,
    double factor_coul, double factor_lj,
    double &fforce)
{
  return 0.0;
}

/* ---------------------------------------------------------------------- */
void PairGranRate::transfer_history(double* sourcevalues, double* targetvalues){
    for (int k = 0; k < size_history; k++){
      targetvalues[k] = sourcevalues[k];
    }
};
