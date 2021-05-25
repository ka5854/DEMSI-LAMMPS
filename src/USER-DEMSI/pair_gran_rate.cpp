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
  
/*
  Allocate memory for the block data according to this block scheme:
  Each element [i] has a block of size N+1 where N is the number of neighbors
  that are "in contact" with [i] as defined by the touching algorithm below.
  The list of contacting neighbors is ls[i][j=0,N] so that ls[i][0]=i, and the
  remaining ls[i][j=1,N] contains the list of up to 6 neighbors, in whatever
  order they may have been found by the touching algorithm.

  There can be at most 6 contacting neighbors so the maximum block size
  is 7.  The array vs[i][j][3] contains the velocity {vx,vy,vz} where the
  azimuthal rotation rate is stored in vz, so the tangential velocity due to
  rotation is radius[j]*vz.
  
  (We use the "auto X = new type[inum]" syntax to create the arrays, and
  assume that the 'delete' that is requited by 'new' is handled automatically.)
*/
/*
  int inum;
  inum = list->inum;
  auto ns = new int[inum]; // number of neighbors "in contact" with [i]
  auto ls = new int[inum][6]; // list of [j] in the [i] block
  auto vs = new double[inum][6][3]; // [i][j] block velocity
*/
}

/* ---------------------------------------------------------------------- */
PairGranRate::~PairGranRate()
{
/*
  int inum;
  inum = list->inum;
  int ns[inum];
  int ls[inum][6];
  double vs[inum][6][3];
  
//delete [] ns;  // compiler issues a 'warning deleting ns'
//delete [] ls;  // and the code segfaults
//delete [] vs;  // so these must be unnecessary (?)
*/
}
/* ---------------------------------------------------------------------- */

void PairGranRate::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

/*
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
*/

  if (int(timeIntegrationFlag) = 0) {
    compute_rate_exverlet();
  } else if(int(timeIntegrationFlag) = 1) {
    compute_rate_explicit(); back_substitute();
  } else if(int(timeIntegrationFlag) = 2) {
    compute_rate_implicit(); back_substitute();
  } else {
    error->all(FLERR,"int(timeIntegrationFlag) must be (0,1,2)");
  }
  
  if (vflag_fdotr) virial_fdotr_compute();

} // end PairGranRate::compute
/* ---------------------------------------------------------------------- */

void PairGranRate::compute_rate_exverlet()
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
  double radi, radj, r, rinv, nx, ny;
  double vrx, vry, vnnr, vtx, vty, wrz, vtrx, vtry;
  double hAVGi, hAVGj, hDeltal;
  double fx, fy, fz;
  double Pplus, Sxplus, Syplus; // outgoing Pressure, ShearStresses
  double SYield;  // shear yield conditions

  int historyupdate = 1;
  if (update->setupflag) historyupdate = 0;

  double dtv = update->dt;
  double dtf = 0.5 * update->dt; // * force->ftm2v;

  /* Load the explicit force/torque */
  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    f[i][0] = 0.0;
    f[i][1] = 0.0;
    torque[i][2] = 0.0;
  }
  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    allhistory = firsthistory[i];
    radi = radius[i];

    // loop over neighbors of each element
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      jtype = atom->type[j];
      j &= NEIGHMASK;
      radj = radius[j];

//    *touch = &firsttouch[i][jj];
      history = &allhistory[size_history*jj];

      /*'history' now points to the ii-jj array that stores
         all the history associated with pair ii-jj
         For all pairs:
         *history[0]: bondPressure
         *history[1]: bondShearX
         *history[2]: bondShearY
         *history[3]: bondFlag [0=no bond, 1=bonded]]
      */

//    delx = x[i][0] - x[j][0]; // hooke form uses inward normal relative to [i]
//    dely = x[i][1] - x[j][1];
      delx = x[j][0] - x[i][0]; // rate form uses outward normal relative to [i]
      dely = x[j][1] - x[i][1];
      rsq = delx*delx + dely*dely;
//    radsum = OverLap*(radius[i] + radius[j]);
      radsum = 1.01*(radi + radj);

      bondFlagIn = int(history[3]);
      Fplus = bondFlagIn;

      if(!bondFlagIn){
        if (rsq >= radsum*radsum){ // no contact
          firsttouch[i][jj] = 0;
          Fplus = 0;
          Pplus = 0.;
          Sxplus = 0.;
          Syplus = 0.;
          fx = fy = fz = 0.;
        } else { // in first contact
          firsttouch[i][jj] = 1;
          Fplus = 1;
        } // end if(rsq >= radsum*radsum
        history[0] = history[1] = history[2] = 0.;
      } // end if(!bondFlagIn)

      if (Fplus){
        r = sqrt(rsq);
        rinv = 1.0/r;
        nx = delx*rinv;
        ny = dely*rinv;

        // relative translational velocity

        vrx = v[j][0] - v[i][0];
        vry = v[j][1] - v[i][1];

        // normal component of the relative translational velocity

        vnnr = vrx*nx + vry*ny;

        // tangential component of the relative translational velocity

        vtx = vrx - vnnr*nx;
        vty = vry - vnnr*ny;

        // relative rotational velocity

        wrz = (radi*omega[i][2] + radj*omega[j][2]);

        // relative tangential velocity

        vtrx = vtx - ny*wrz;
        vtry = vty + nx*wrz;

        Pplus  = history[0] - bulkModulus*dtv*vnnr*rinv;
        Sxplus = history[1] -shearModulus*dtv*vtrx*rinv;
        Syplus = history[2] -shearModulus*dtv*vtry*rinv;

        hAVGi = rmass[i]/(rho0*MY_PI*radi*radi);
        hAVGj = rmass[j]/(rho0*MY_PI*radj*radj);

        hDeltal = 2*MY_PI3*(hAVGi*radi * hAVGj*radj)
                          /(hAVGi*radi + hAVGj*radj);

        fx = (Pplus*nx + Sxplus)*hDeltal;
        fy = (Pplus*ny + Syplus)*hDeltal;
        fz = (Sxplus*ny - Syplus*nx)*hDeltal;
        
        f[i][0] -= fx;
        f[i][1] -= fy;
        torque[i][2] -= radi*fz;

        if (force->newton_pair || j < atom->nlocal){
          f[j][0] += fx;
          f[j][1] += fy;
          torque[j][2] -= radj*fz;
        }

          // test for tensile fracture, compressive flowstress, shear flowstress

          if (Pplus <= tensileFractureStress) { // tensile fracture
            Pplus = 0.;
            Fplus = 0;
          } else if (Pplus > compressiveYieldStress) { // perfectly plastic compressive flowstress
            Pplus = compressiveYieldStress;
          }
          SYield = sqrt(Sxplus*Sxplus + Syplus*Syplus)/shearYieldStress;
          if (SYield > 1.0) {
//          Sxplus /= SYield;
//          Syplus /= SYield;
            Sxplus = Syplus = 0.;
          }

          // update history
          if(historyupdate){
            history[0] = Pplus;
            history[1] = Sxplus;
            history[2] = Syplus;
            history[3] = double(Fplus);
          }

      } // end if (Fplus)

      if (evflag) ev_tally_xyz(i,j,atom->nlocal, force->newton_pair,
              0.0,0.0,fx,fy,0,x[i][0]-x[j][0],x[i][1]-x[j][1],0);
              

    } // end for (jj = 0; jj < jnum; jj++)
  } // end for (ii = 0; ii < inum; ii++)
} // end PairGranRate::compute_rate_verlet
/* ---------------------------------------------------------------------- */

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
  double radi, radj, r, rinv, nx, ny;
  double vrx, vry, vnnr, vtx, vty, wrz, vtrx, vtry;
  double hAVGi, hAVGj, hDeltal;
  double fx, fy, fz;
  double Pplus, Sxplus, Syplus; // outgoing Pressure, ShearStresses
  double SYield;  // shear yield conditions

  int historyupdate = 1;
  if (update->setupflag) historyupdate = 0;

  double dtv = update->dt;
  double dtf = 0.5 * update->dt; // * force->ftm2v;

  /* Load the explicit force/torque */
  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    f[i][0] = 0.0;
    f[i][1] = 0.0;
    torque[i][2] = 0.0;
  }
  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    allhistory = firsthistory[i];
    radi = radius[i];

    // loop over neighbors of each element
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      jtype = atom->type[j];
      j &= NEIGHMASK;
      radj = radius[j];

//    *touch = &firsttouch[i][jj];
      history = &allhistory[size_history*jj];

      /*'history' now points to the ii-jj array that stores
         all the history associated with pair ii-jj
         For all pairs:
         *history[0]: bondPressure
         *history[1]: bondShearX
         *history[2]: bondShearY
         *history[3]: bondFlag [0=no bond, 1=bonded]]
      */

//    delx = x[i][0] - x[j][0]; // hooke form uses inward normal relative to [i]
//    dely = x[i][1] - x[j][1];
      delx = x[j][0] - x[i][0]; // rate form uses outward normal relative to [i]
      dely = x[j][1] - x[i][1];
      rsq = delx*delx + dely*dely;
//    radsum = OverLap*(radius[i] + radius[j]);
      radsum = 1.01*(radi + radj);

      bondFlagIn = int(history[3]);
      Fplus = bondFlagIn;

      if(!bondFlagIn){
        if (rsq >= radsum*radsum){ // no contact
          firsttouch[i][jj] = 0;
          Fplus = 0;
          Pplus = 0.;
          Sxplus = 0.;
          Syplus = 0.;
          fx = fy = fz = 0.;
        } else { // in first contact
          firsttouch[i][jj] = 1;
          Fplus = 1;
        } // end if(rsq >= radsum*radsum
        history[0] = history[1] = history[2] = 0.;
      } // end if(!bondFlagIn)

      if (Fplus){
        r = sqrt(rsq);
        rinv = 1.0/r;
        nx = delx*rinv;
        ny = dely*rinv;

        // relative translational velocity

        vrx = v[j][0] - v[i][0];
        vry = v[j][1] - v[i][1];

        // normal component of the relative translational velocity

        vnnr = vrx*nx + vry*ny;

        // tangential component of the relative translational velocity

        vtx = vrx - vnnr*nx;
        vty = vry - vnnr*ny;

        // relative rotational velocity

        wrz = (radi*omega[i][2] + radj*omega[j][2]);

        // relative tangential velocity

        vtrx = vtx - ny*wrz;
        vtry = vty + nx*wrz;

        Pplus  =  -bulkModulus*dtf*vnnr*rinv;
        Sxplus = -shearModulus*dtf*vtrx*rinv;
        Syplus = -shearModulus*dtf*vtry*rinv;

        hAVGi = rmass[i]/(rho0*MY_PI*radi*radi);
        hAVGj = rmass[j]/(rho0*MY_PI*radj*radj);

        hDeltal = 2*MY_PI3*(hAVGi*radi * hAVGj*radj)
                          /(hAVGi*radi + hAVGj*radj);

//      fx = Pplus*nx*hDeltal;
//      fy = Pplus*ny*hDeltal;
        fx = (Pplus*nx + Sxplus)*hDeltal;
        fy = (Pplus*ny + Syplus)*hDeltal;
        fz = (Sxplus*ny - Syplus*nx)*hDeltal;
        
        f[i][0] -= fx;
        f[i][1] -= fy;
        torque[i][2] -= radi*fz;

        if (force->newton_pair || j < atom->nlocal){
          f[j][0] += fx;
          f[j][1] += fy;
          torque[j][2] -= radj*fz;
        }

      } // end if (Fplus)
    } // end for (jj = 0; jj < jnum; jj++)
  } // end for (ii = 0; ii < inum; ii++)

  /*! Put the new velocity into vn for fix_nve_sphere_demsi::integrate_final.
      Recall that fix_nve_sphere_demsi:integrate_initial has put the explicit
      part of the update into {v[0,1],omega[2]}. */
  for (int i = 0; i < nlocal; i++) {
//  if (mask[i] & groupbit) {

      vn[i][0] =     v[i][0] + dtv*     f[i][0]/rmass[i];
      vn[i][1] =     v[i][1] + dtv*     f[i][1]/rmass[i];
      vn[i][2] = omega[i][2] + dtv*torque[i][2]/(0.5*rmass[i]*radi*radi);

//  }
  } // end for (int i = 0; i < nlocal; i++)
} // end PairGranRate::compute_rate_explicit
/* ---------------------------------------------------------------------- */

void PairGranRate::compute_rate_implicit()
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
 
  auto ns = new int[inum]; // number of neighbors "in contact" with [i]
  auto ls = new int[inum][6]; // list of [j] in the [i] block
  auto vs = new double[inum][6][3]; // [i][j] block velocity
 
//if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  int bondFlagIn; // incoming bondFlag
  int Fplus;      // outgoing bondFlag

  double delx, dely, rsq, radsum;
  double radi, radj, r, rinv, nx, ny;
  double vrx, vry, vnnr, vtx, vty, wrz, vtrx, vtry;
  double hAVGi, hAVGj, hDeltal;
  double fx, fy, fz;
  double Pplus, Sxplus, Syplus; // outgoing Pressure, ShearStresses
  double SYield;  // shear yield conditions

  int historyupdate = 1;
  if (update->setupflag) historyupdate = 0;

  double dtv = update->dt;
  double dtf = 0.5 * update->dt; // * force->ftm2v;

  /* Load the explicit force/torque */
  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    f[i][0] = 0.0;
    f[i][1] = 0.0;
    torque[i][2] = 0.0;
  }
  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    allhistory = firsthistory[i];
    radi = radius[i];

    // loop over neighbors of each element
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      jtype = atom->type[j];
      j &= NEIGHMASK;
      radj = radius[j];

//    *touch = &firsttouch[i][jj];
      history = &allhistory[size_history*jj];

      /*'history' now points to the ii-jj array that stores
         all the history associated with pair ii-jj
         For all pairs:
         *history[0]: bondPressure
         *history[1]: bondShearX
         *history[2]: bondShearY
         *history[3]: bondFlag [0=no bond, 1=bonded]]
      */

//    delx = x[i][0] - x[j][0]; // hooke form uses inward normal relative to [i]
//    dely = x[i][1] - x[j][1];
      delx = x[j][0] - x[i][0]; // rate form uses outward normal relative to [i]
      dely = x[j][1] - x[i][1];
      rsq = delx*delx + dely*dely;
//    radsum = OverLap*(radius[i] + radius[j]);
      radsum = 1.01*(radi + radj);

      bondFlagIn = int(history[3]);
      Fplus = bondFlagIn;

      if(!bondFlagIn){
        if (rsq >= radsum*radsum){ // no contact
          firsttouch[i][jj] = 0;
          Fplus = 0;
          Pplus = 0.;
          Sxplus = 0.;
          Syplus = 0.;
          fx = fy = fz = 0.;
        } else { // in first contact
          firsttouch[i][jj] = 1;
          Fplus = 1;
        } // end if(rsq >= radsum*radsum
        history[0] = history[1] = history[2] = 0.;
      } // end if(!bondFlagIn)

      if (Fplus){
        r = sqrt(rsq);
        rinv = 1.0/r;
        nx = delx*rinv;
        ny = dely*rinv;

        // relative translational velocity

        vrx = v[j][0] - v[i][0];
        vry = v[j][1] - v[i][1];

        // normal component of the relative translational velocity

        vnnr = vrx*nx + vry*ny;

        // tangential component of the relative translational velocity

        vtx = vrx - vnnr*nx;
        vty = vry - vnnr*ny;

        // relative rotational velocity

        wrz = (radi*omega[i][2] + radj*omega[j][2]);

        // relative tangential velocity

        vtrx = vtx - ny*wrz;
        vtry = vty + nx*wrz;

        Pplus  =  -bulkModulus*dtf*vnnr*rinv;
        Sxplus = -shearModulus*dtf*vtrx*rinv;
        Syplus = -shearModulus*dtf*vtry*rinv;

        hAVGi = rmass[i]/(rho0*MY_PI*radi*radi);
        hAVGj = rmass[j]/(rho0*MY_PI*radj*radj);

        hDeltal = 2*MY_PI3*(hAVGi*radi * hAVGj*radj)
                          /(hAVGi*radi + hAVGj*radj);

//      fx = Pplus*nx*hDeltal;
//      fy = Pplus*ny*hDeltal;
        fx = (Pplus*nx + Sxplus)*hDeltal;
        fy = (Pplus*ny + Syplus)*hDeltal;
        fz = (Sxplus*ny - Syplus*nx)*hDeltal;
        
        f[i][0] -= fx;
        f[i][1] -= fy;
        torque[i][2] -= radi*fz;

        if (force->newton_pair || j < atom->nlocal){
          f[j][0] += fx;
          f[j][1] += fy;
          torque[j][2] -= radj*fz;
        }

      } // end if (Fplus)
    } // end for (jj = 0; jj < jnum; jj++)
  } // end for (ii = 0; ii < inum; ii++)

  /*! Put the new velocity into vn for fix_nve_sphere_demsi::integrate_final.
      Recall that fix_nve_sphere_demsi:integrate_initial has put the explicit
      part of the update into {v[0,1],omega[2]}. */
  for (int i = 0; i < nlocal; i++) {
//  if (mask[i] & groupbit) {

      vn[i][0] =     v[i][0] + dtv*     f[i][0]/rmass[i];
      vn[i][1] =     v[i][1] + dtv*     f[i][1]/rmass[i];
      vn[i][2] = omega[i][2] + dtv*torque[i][2]/(0.5*rmass[i]*radi*radi);

//  }
  } // end for (int i = 0; i < nlocal; i++)
/*
  Just for practice, load ns and the arrays ls and vs.
*/

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    allhistory = firsthistory[i];
    
    ls[ii][0] = i;
    ns[ii] = 1; // number of elements in the block
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      history = &allhistory[size_history*jj];
      
      if (history[3]) { // the neighbor is "in contact"
        if(ns[ii]+1 > 7) break; // should never happen, but we need to check...
        ls[ii][jj] = j;
        vs[ii][jj][0] = vn[j][0]; // or v[j][0];
        vs[ii][jj][1] = vn[j][1]; // or v[j][1];
        vs[ii][jj][2] = vn[j][2]; // or omega[j][2];
        ns[ii] += 1;
      }
    } // end for (jj = 0; jj < jnum; jj++)
  } // end for (ii = 0; ii < inum; ii++)

  delete[] ns;
  delete[] ls;
  delete[] vs;
  
} // end PairGranRate::compute_rate_implicit
/* ---------------------------------------------------------------------- */

void PairGranRate::back_substitute()
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
  double radi, radj, r, rinv, nx, ny;
  double vrx, vry, vnnr, vtx, vty, wrz, vtrx, vtry;
  double hAVGi, hAVGj, hDeltal;
  double fx, fy, fz;
  double Pplus, Sxplus, Syplus; // outgoing Pressure, ShearStresses
  double SYield;  // shear yield conditions

  int historyupdate = 1;
  if (update->setupflag) historyupdate = 0;

  double dtv = update->dt;
  double dtf = 0.5 * update->dt; // * force->ftm2v;

  /* Load the force/torque */

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    f[i][0] = 0.0;
    f[i][1] = 0.0;
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
    radi = radius[i];

    // loop over neighbors of each element
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      jtype = atom->type[j];
      j &= NEIGHMASK;
      radj = radius[j];

//    *touch = &firsttouch[i][jj];
      history = &allhistory[size_history*jj];

      /*'history' now points to the ii-jj array that stores
         all the history associated with pair ii-jj
         For all pairs:
         *history[0]: bondPressure
         *history[1]: bondShearX
         *history[2]: bondShearY
         *history[3]: bondFlag [0=no bond, 1=bonded]]
      */
      
//    delx = x[i][0] - x[j][0]; // hopkins uses inward normal relative to [i]
//    dely = x[i][1] - x[j][1];
      delx = x[j][0] - x[i][0]; // rateform uses outward normal relative to [i] ??
      dely = x[j][1] - x[i][1];
      rsq = delx*delx + dely*dely;
//    radsum = OverLap*(radius[i] + radius[j]);
      radsum = 1.01*(radi + radj);

      bondFlagIn = int(history[3]);
      Fplus = bondFlagIn;

      if(!bondFlagIn){
        if (rsq >= radsum*radsum){ // no contact
          firsttouch[i][jj] = 0;
          Fplus = 0;
          Pplus = 0.;
          Sxplus = 0.;
          Syplus = 0.;
          fx = fy = fz = 0.;
        } else { // in first contact
          firsttouch[i][jj] = 1;
          Fplus = 1;
        } // end if(rsq >= radsum*radsum
        history[0] = history[1] = history[2] = 0.;
      } // end if(!bondFlagIn)

      if(Fplus){
        r = sqrt(rsq);
        rinv = 1.0/r;
        nx = delx*rinv;
        ny = dely*rinv;

        // relative translational velocity

        vrx = vn[j][0] - vn[i][0];
        vry = vn[j][1] - vn[i][1];

        // normal component of the relative translational velocity

        vnnr = vrx*nx + vry*ny;

        // tangential component of the relative translational velocity

        vtx = vrx - vnnr*nx;
        vty = vry - vnnr*ny;

        // relative rotational velocity

        wrz = (radi*omega[i][2] + radj*omega[j][2]);

        // relative tangential velocity

        vtrx = vtx - ny*wrz;
        vtry = vty + nx*wrz;

        Pplus  = history[0]  - bulkModulus*dtf*vnnr*rinv;
        Sxplus = history[1] - shearModulus*dtf*vtrx*rinv;
        Syplus = history[2] - shearModulus*dtf*vtry*rinv;

      /*! Reload {f,torque} for the next integrate_initial */

        hAVGi = rmass[i]/(rho0*MY_PI*radi*radi);
        hAVGj = rmass[j]/(rho0*MY_PI*radj*radj);

        hDeltal = 2*MY_PI3*(hAVGi*radi * hAVGj*radj)
                          /(hAVGi*radi + hAVGj*radj);

//      fx = Pplus*nx*hDeltal;
//      fy = Pplus*ny*hDeltal;
        fx = (Pplus*nx + Sxplus)*hDeltal;
        fy = (Pplus*ny + Syplus)*hDeltal;
        fz = (Sxplus*ny - Syplus*nx)*hDeltal;

        f[i][0] -= fx;
        f[i][1] -= fy;
        torque[i][2] -= radi*fz;

        if (force->newton_pair || j < atom->nlocal){
          f[j][0] += fx;
          f[j][1] += fy;
          torque[j][2] -= radj*fz;
        }

        // test for tensile fracture, compressive flowstress, shear flowstress

        if (Pplus <= tensileFractureStress) { // tensile fracture
          Pplus = 0.;
          Fplus = 0;
        } else if (Pplus > compressiveYieldStress) { // perfectly plastic compressive flowstress
          Pplus = compressiveYieldStress;
        }
        SYield = sqrt(Sxplus*Sxplus + Syplus*Syplus)/shearYieldStress;
        if (SYield > 1.0) {
//          Sxplus /= SYield;
//          Syplus /= SYield;
            Sxplus = Syplus = 0.;
        }
      } // end if(Fplus)
      
      // update history
      if(historyupdate){
        history[0] = Pplus;
        history[1] = Sxplus;
        history[2] = Syplus;
        history[3] = double(Fplus);
      }

      if (evflag) ev_tally_xyz(i,j,atom->nlocal, force->newton_pair,
              0.0,0.0,fx,fy,0,x[i][0]-x[j][0],x[i][1]-x[j][1],0);

    } // end for (ii = 0; ii < inum; ii++)
  } // end for (ii = 0; ii < inum; ii++)
} // end PairGranRate::back_substitute()

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
