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

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairGranRate::PairGranRate(LAMMPS *lmp) :
              PairGranHookeHistory(lmp, 12)
{
  nondefault_history_transfer = 1;
  beyond_contact = 1; // this is a pair thing, that I don't quite understand...

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

  if (int(timeIntegrationFlag) == 0) {
    compute_rate_exverlet();
  } else if(int(timeIntegrationFlag) == 1) {
    compute_rate_explicit(); load_new_forces();
  } else if(int(timeIntegrationFlag) == 2) {
    compute_rate_implicit(); load_new_forces();
  } else {
    error->all(FLERR,"int(timeIntegrationFlag) must be (0,1,2)");
  }

  if (vflag_fdotr) virial_fdotr_compute();

} // end PairGranRate::compute
/* ---------------------------------------------------------------------- */

void PairGranRate::compute_rate_exverlet() {
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
  double *ice_area = atom->ice_area;
  double *mean_thickness = atom->mean_thickness;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int bondFlagIn; // incoming bondFlag
  int Fplus;      // outgoing bondFlag

  double delx, dely, rsq, radsum;
  double radi, radj, r, rinv, nx, ny, tx, ty;
  double vrx, vry, vnnr, wrz, vttr;
  double hAVGi, hAVGj, hDeltal;
  double fx, fy, fz;
  double Pplus, Splus, Sxplus, Syplus; // outgoing Pressure, ShearStresses
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
         *history[1]: bondShear
         *history[2]: bondFlag [0=no bond, 1=bonded]]
      */

      delx = x[j][0] - x[i][0]; // outward normal relative to [i]
      dely = x[j][1] - x[i][1];
      rsq = delx*delx + dely*dely;
      radsum = 1.10*(radi + radj);

      bondFlagIn = int(history[2]);
      Fplus = bondFlagIn;

      if(!bondFlagIn){
        if (rsq >= radsum*radsum){ // no contact
          firsttouch[i][jj] = 0;
          Fplus = 0;
          Pplus = 0.;
          Splus = 0.;
          fx = fy = fz = 0.;
        } else { // in first contact
          firsttouch[i][jj] = 1;
          Fplus = 1;
        } // end if(rsq >= radsum*radsum
        history[0] = history[1] = 0.;
      } // end if(!bondFlagIn)

        r = sqrt(rsq);
        nx = delx/r;
        ny = dely/r;
        tx = -ny;
        ty =  nx;
        rinv = 1.0/r;

        // relative translational velocity vector
//      vrx = v[j][0] - v[i][0];
//      vry = v[j][1] - v[i][1];
        vrx = vn[j][0] - vn[i][0];
        vry = vn[j][1] - vn[i][1];

        // relative rotational velocity
        double radij = 2*(radi * radj)/(radi + radj); // Harmonic mean radius
//      wrz = -radij*(omega[i][2] + omega[j][2]);
        wrz = -radij*(vn[i][2] + vn[j][2]);

        // magnitude of the normal relative velocity
        vnnr = vrx*nx + vry*ny;

        // magnitude of the relative tangential velocity
        vttr = vrx*tx + vry*ty + wrz;

      if (Fplus){
        Pplus = history[0]  + bulkModulus*dtv*vnnr*rinv; // full step stress
        Splus = history[1] + shearModulus*dtv*vttr*rinv;
      } // end if (Fplus)

        Sxplus = Splus*tx;
        Syplus = Splus*ty;

        // test for tensile fracture, compressive flowstress, shear flowstress
//      if (jtype != 2) { // do not limit stresses on the coastline (~noslip boundary condition)
          if (Pplus >= tensileFractureStress) { // tensile fracture
            Pplus = 0.;
            Splus = 0.; // maybe
            Fplus = 0;
            v[i][2] += 1.;
            v[j][2] += 1.;
          } else if (Pplus < compressiveYieldStress) { // perfectly plastic compressive flowstress
            Pplus = compressiveYieldStress;
            omega[i][0] += 1.;
            omega[j][0] += 1.;
          }
          SYield = sqrt(Sxplus*Sxplus + Syplus*Syplus)/shearYieldStress;
          if (SYield > 1.0) { // perfectly plastic shear flowstress
            Splus /= SYield;
            omega[i][1] += 1.;
            omega[j][1] += 1.;
          }
//      } // end if (jtype != 2)

        // update history
        if(historyupdate){
          history[0] = Pplus;
          history[1] = Splus;
          history[2] = double(Fplus);
        }

        hAVGi = mean_thickness[i];
        hAVGj = mean_thickness[j];
        hDeltal = 2.*MY_PI3*(radi*hAVGi * radj*hAVGj)
                           /(radi*hAVGi + radj*hAVGj); // contact Area

        fx = (Pplus*nx + Splus*tx)*hDeltal;
        fy = (Pplus*ny + Splus*ty)*hDeltal;
        fz = -radij*Splus*hDeltal;

        f[i][0] += fx;
        f[i][1] += fy;
        torque[i][2] += fz;

        if (force->newton_pair || j < atom->nlocal){
          f[j][0] -= fx;
          f[j][1] -= fy;
          torque[j][2] += fz;
        }
        
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
  double *ice_area = atom->ice_area;
  double *mean_thickness = atom->mean_thickness;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int bondFlagIn; // incoming bondFlag
  int Fplus;      // outgoing bondFlag

  double delx, dely, rsq, radsum;
  double radi, radj, r, rinv, nx, ny, tx, ty;
  double vrx, vry, vnnr, wrz, vttr;
  double hAVGi, hAVGj, hDeltal;
  double fx, fy, fz;
  double Pplus, Splus, Sxplus, Syplus; // outgoing Pressure, ShearStresses
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
         *history[1]: bondShear
         *history[2]: bondFlag [0=no bond, 1=bonded]]
      */

      delx = x[j][0] - x[i][0]; // outward normal relative to [i]
      dely = x[j][1] - x[i][1];
      rsq = delx*delx + dely*dely;
      radsum = 1.10*(radi + radj);

      bondFlagIn = int(history[2]);
      Fplus = bondFlagIn;

      if(!Fplus) continue;
/*
      if(!bondFlagIn){
        if (rsq >= radsum*radsum){ // no contact
          firsttouch[i][jj] = 0;
          Fplus = 0;
          Pplus = 0.;
          Splus = 0.;
          fx = fy = fz = 0.;
        } else { // in first contact
          firsttouch[i][jj] = 1;
          Fplus = 1;
        } // end if(rsq >= radsum*radsum
        history[0] = history[1] = 0.;
      } // end if(!bondFlagIn)
*/

//    if (Fplus){
        r = sqrt(rsq);
        nx = delx/r;
        ny = dely/r;
        tx = -ny;
        ty =  nx;
        rinv = 1.0/r;

        // relative translational velocity vector
        vrx = vn[j][0] - vn[i][0];
        vry = vn[j][1] - vn[i][1];

        // relative rotational velocity
        double radij = 2*(radi * radj)/(radi + radj); // Harmonic mean radius
        wrz = -radij*(vn[i][2] + vn[j][2]);

        // magnitude of the normal relative velocity
        vnnr = vrx*nx + vry*ny;

        // magnitude of the relative tangential velocity
        vttr = vrx*tx + vry*ty + wrz;

      if (Fplus){
        Pplus =  bulkModulus*dtf*vnnr*rinv; // half step stress increment
        Splus = shearModulus*dtf*vttr*rinv;
      } else {
//        double rbyr = radsum*rinv;
//        Pplus =  - bulkModulus*dtv*vnnr*rinv*rbyr*rbyr;
//        Splus = - shearModulus*dtv*vttr*rinv*rbyr*rbyr;
      }
      
        Sxplus = Splus*tx;
        Syplus = Splus*ty;
        
        hAVGi = mean_thickness[i];
        hAVGj = mean_thickness[j];
        hDeltal = 2.*MY_PI3*(radi*hAVGi * radj*hAVGj)
                           /(radi*hAVGi + radj*hAVGj); // contact Area

        fx = (Pplus*nx + Sxplus)*hDeltal;
        fy = (Pplus*ny + Syplus)*hDeltal;
        fz = -radij*Splus*hDeltal;

        f[i][0] += fx;
        f[i][1] += fy;
        torque[i][2] += fz;

        if (force->newton_pair || j < atom->nlocal){
          f[j][0] -= fx;
          f[j][1] -= fy;
          torque[j][2] += fz;
        }
//    } // end if (Fplus)
    } // end for (jj = 0; jj < jnum; jj++)
  } // end for (ii = 0; ii < inum; ii++)

  // full step with the half step stress increment
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
//for (int i = 0; i < nlocal; i++) {
    itype = atom->type[i];
    if (itype != 1) continue;
      double dtm = dtv/rmass[i];
      radi = radius[i];
      vn[i][0] =     v[i][0] + dtm*     f[i][0]/rmass[i];
      vn[i][1] =     v[i][1] + dtm*     f[i][1]/rmass[i];
      vn[i][2] = omega[i][2]; // + dtm*torque[i][2]/(0.5*radi*radi);
  } // end for (int i = 0; i < nlocal; i++)
} // end PairGranRate::compute_rate_explicit
/* ---------------------------------------------------------------------- */

void PairGranRate::compute_rate_implicit()
{
  int i,j,ii,jj,inum,jnum;
  int k,l;
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
  double *ice_area = atom->ice_area;
  double *mean_thickness = atom->mean_thickness;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int bondFlagIn; // incoming bondFlag
  int Fplus;      // outgoing bondFlag

  double delx, dely, rsq, radsum;
  double radi, radj, r, rinv, nx, ny, tx, ty;
  double vrx, vry, vnnr, wrz, vttr;
  double hAVGi, hAVGj, hDeltal;
  double fx, fy, fz;
  double Pplus, Splus, Sxplus, Syplus; // outgoing Pressure, ShearStresses
  double SYield;  // shear yield conditions

  int historyupdate = 1;
  if (update->setupflag) historyupdate = 0;

  double dtv = update->dt;
  double dtf = 0.5 * update->dt; // * force->ftm2v;

/*
  Allocate memory for the block data according to this blocking scheme:
  Each element [i] has a block of size N where N is the number of neighbors
  that are "in contact" with [i] as defined by the touching algorithm below.

  The list of contacting neighbors is ls[i][n=0,N] so that ls[i][0]=i, and the
  remaining ls[i][n=1,N] contains the list of up to 6 neighbors, in whatever
  order they may have been found by the touching algorithm.

  The array vs[i][n][3] contains the velocity {vx,vy,vz} where the
  azimuthal rotation rate is stored in vz, so the tangential velocity due to
  rotation is radius[j]*vz.
*/
  const int nmax = 13; // 1 + max number of neighbors
  auto ns = new int[inum]; // N = number of neighbors "in contact" with [i]
  auto ls = new int[inum][nmax]; // list of [j] in the [i] block
  auto vs = new double[inum][nmax][3]; // [i][j] block velocity

/* Load the implicit accelerations by solving the system x=Inverse[A].b for
   the new velocity x such that x-vn = Delta[v]. */
  const int ndim = 3;
/* work space: uses indices starting with 1 */
  const int wdim = (1 + nmax)*ndim;
  double A[wdim][wdim];  // upper echelon matrix of coefficients
  double P[wdim][wdim];  // the full matrix of pivots
  double b[wdim];        // the right-side vector, and solution vector

/* load the lists ns, ls, vs */
  int nsp;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    allhistory = firsthistory[i];
    
    ns[ii] = 0; // pair number at ii, so far...
    ls[ii][0] = i; // element number at ii
    vs[ii][0][0] = v[i][0];
    vs[ii][0][1] = v[i][1];
    vs[ii][0][2] = omega[i][2];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
//    jtype = atom->type[j];
//    if (jtype != 1) continue; // removes coastline type from the system
      history = &allhistory[size_history*jj];
      if (history[2]) { // the neighbor is "in contact"
        nsp = ns[ii] + 1;
        if(nsp > nmax-1) break; // should never happen, but we need to check...
        ns[ii] = nsp;
        ls[ii][nsp] = j;
        vs[ii][nsp][0] = v[j][0];
        vs[ii][nsp][1] = v[j][1];
        vs[ii][nsp][2] = omega[j][2];
      } // end if (history[2])
    } // end for (jj = 0; jj < jnum; jj++)
  } // end for (ii = 0; ii < inum; ii++)

  // set up and solve for the implicit acceleration
  for (ii = 0; ii < inum; ii++) if (ns[ii] > 0) {
    i = ilist[ii];
    int mf = 1 + ns[ii]; // number of vector equations == number in contact at ii
    
    // load the rhs vector b
    for (int kf = 1; kf <= mf; kf++){
      for (int jd = 1; jd <= ndim; jd++){
        int neq_k = mf*(jd-1) + kf;
        b[neq_k] = vs[ii][kf-1][jd-1];
      }
    }

    // load the matrix A
    for (int lf = 2; lf <= mf; lf++){ // column
      for (int kf = 1; kf <= lf-1; kf++){ // row
        for (int jd = 1; jd <= ndim; jd++){
          int neq_k = mf*(jd-1) + kf;
          int neq_l = mf*(jd-1) + lf;
          for (int id = 1; id <= ndim; id++){
            int ind_kf = mf*(id-1) + kf;
            int ind_lf = mf*(id-1) + lf;
            A[neq_k][ind_kf] = A[neq_l][ind_kf] =
            A[neq_k][ind_lf] = A[neq_l][ind_lf] = 0.0;
          }
        }
      }
    }

    for (int lf = 2; lf <= mf; lf++){ // column
      for (int kf = 1; kf <= lf-1; kf++){ // row
        i = ls[ii][kf-1];
        j = ls[ii][lf-1];
        radi = radius[i];
        radj = radius[j];
        delx = x[j][0] - x[i][0]; // outward normal relative to [i]
        dely = x[j][1] - x[i][1];
        rsq = delx*delx + dely*dely;

        r = sqrt(rsq);
        nx = delx/r;
        ny = dely/r;
        tx = -ny;
        ty =  nx;
        rinv = 1.0/r;

        hAVGi = mean_thickness[i];
        hAVGj = mean_thickness[j];
        hDeltal = 2.*MY_PI3*(radi*hAVGi * radj*hAVGj)
                           /(radi*hAVGi + radj*hAVGj); // contact Area

        double   dtfac = (dtv*dtf)*hDeltal*rinv;
        double  dtbulk =  bulkModulus*dtfac;
        double dtshear = shearModulus*dtfac;
        double  kappaK =  (dtbulk/rmass[i]);
        double  kappaL =  (dtbulk/rmass[j]);
        double  gammaK = (dtshear/rmass[i]);
        double  gammaL = (dtshear/rmass[j]);

// 2d form works okay-ish; has large shear yields at free edges of the sea ice.
//      double nn[2][2] = {{nx*nx,nx*ny},{ny*nx,ny*ny}};
//      double tt[2][2] = {{tx*tx,tx*ty},{ty*tx,ty*ty}};

// fully coupled form fails with really bad omega.

        double Rij = 2.*(radi*radj)/(radi+radj);
        double  GK = 2.*Rij/(radi*radi);
        double  GL = 2.*Rij/(radj*radj);
        double  nn[3][3] = {{nx*nx,nx*ny,     0.},{ny*nx,ny*ny,     0.},{   0.,   0.,     0.}};
//      double ttK[3][3] = {{tx*tx,tx*ty,-Rij*tx},{ty*tx,ty*ty,-Rij*ty},{-GK*tx,-GK*ty, GK*Rij}};
//      double tmK[3][3] = {{tx*tx,tx*ty, Rij*tx},{ty*tx,ty*ty, Rij*ty},{-GK*tx,-GK*ty,-GK*Rij}};
//      double ttL[3][3] = {{tx*tx,tx*ty,-Rij*tx},{ty*tx,ty*ty,-Rij*ty},{-GL*tx,-GL*ty, GL*Rij}};
//      double tmL[3][3] = {{tx*tx,tx*ty, Rij*tx},{ty*tx,ty*ty, Rij*ty},{-GL*tx,-GL*ty,-GL*Rij}};
        double ttK[3][3] = {{tx*tx,tx*ty, Rij*tx},{ty*tx,ty*ty, Rij*ty},{-GK*tx,-GK*ty,-GK*Rij}};
        double tmK[3][3] = {{tx*tx,tx*ty,-Rij*tx},{ty*tx,ty*ty,-Rij*ty},{-GK*tx,-GK*ty, GK*Rij}};
        double ttL[3][3] = {{tx*tx,tx*ty, Rij*tx},{ty*tx,ty*ty, Rij*ty},{-GL*tx,-GL*ty,-GL*Rij}};
        double tmL[3][3] = {{tx*tx,tx*ty,-Rij*tx},{ty*tx,ty*ty,-Rij*ty},{-GL*tx,-GL*ty, GL*Rij}};

        for (int jd = 1; jd <= ndim; jd++){
          int neq_k = mf*(jd-1) + kf;
          int neq_l = mf*(jd-1) + lf;
          for (int id = 1; id <= ndim; id++){
            int ind_kf = mf*(id-1) + kf;
            int ind_lf = mf*(id-1) + lf;
/*
            A[neq_k][ind_kf] = A[neq_k][ind_kf] + (kappaK*nn[id-1][jd-1] + gammaK*tt[id-1][jd-1]);
            A[neq_k][ind_lf] = A[neq_k][ind_lf] - (kappaK*nn[id-1][jd-1] + gammaK*tt[id-1][jd-1]);
            A[neq_l][ind_lf] = A[neq_l][ind_lf] + (kappaL*nn[id-1][jd-1] + gammaL*tt[id-1][jd-1]);
            A[neq_l][ind_kf] = A[neq_l][ind_kf] - (kappaL*nn[id-1][jd-1] + gammaL*tt[id-1][jd-1]);
*/
            A[neq_k][ind_kf] = A[neq_k][ind_kf] + (kappaK*nn[id-1][jd-1] + gammaK*ttK[id-1][jd-1]);
            A[neq_k][ind_lf] = A[neq_k][ind_lf] - (kappaK*nn[id-1][jd-1] + gammaK*tmK[id-1][jd-1]);
            A[neq_l][ind_lf] = A[neq_l][ind_lf] + (kappaL*nn[id-1][jd-1] + gammaL*ttL[id-1][jd-1]);
            A[neq_l][ind_kf] = A[neq_l][ind_kf] - (kappaL*nn[id-1][jd-1] + gammaL*tmL[id-1][jd-1]);

          }
        }
      }
    }

  // forward elimination, with a special diagonal calculation
  A[1][1] = A[1][1] + 1.0;
  for (j = 2; j <= mf*ndim; j++) { // elimination step
    for (k = j; k <= mf*ndim; k++) { // row
      A[k][k] = 1.0;
      P[k][j-1] = A[k][j-1]/A[j-1][j-1];
      A[k][j-1] = P[k][j-1];
      for (int l = 1; l <= j-2; l++){ // column
        A[k][l] = A[k][l] - P[k][j-1]*A[j-1][l];
        A[k][k] = A[k][k] - A[k][l];
      }
      A[k][k] = A[k][k] - A[k][j-1];
      for (int l = j; l <= k-1; l++){ // column
        A[k][l] = A[k][l] - P[k][j-1]*A[j-1][l];
        A[k][k] = A[k][k] - A[k][l];
      }
      for (int l = k+1; l <= mf*ndim; l++){ // column
        A[k][l] = A[k][l] - P[k][j-1]*A[j-1][l];
        A[k][k] = A[k][k] - A[k][l];
      }
      b[k] = b[k] - P[k][j-1]*b[j-1];
    }
  }

  // back substitution
  for (j = mf*ndim; j > 0; j--){
    for (k = j+1; k <= mf*ndim; k++) {
      b[j] = b[j] - A[j][k]*b[k];
    }
    b[j] = b[j]/A[j][j];
  }

    // load Delta[v]
    for (int kf = 1; kf <= mf; kf++){
      i = ls[ii][kf-1];
      for (int jd = 1; jd <= ndim; jd++){
        int neq_k = mf*(jd-1) + kf;
          vs[ii][kf-1][jd-1] = b[neq_k] - v[i][jd-1];
//        vs[ii][kf-1][jd-1] = b[neq_k] - vn[i][jd-1];
      }
    }
  } // end for (ii = 0; ii < inum; ii++) if(ns[ii] > 0) {

// here we need to communicate ns,ls,vs ... otherwise, just load vn ???

  // get the averaged time(n+1) v; and load it into vn
  
  double mvx[inum], mvy[inum], mvz[inum], m0t[inum], mIt[inum];
  for (ii = 0; ii < inum; ii++) {
    i = ls[ii][0];
    mvx[i] = mvy[i] = mvz[i] = m0t[i] = mIt[i] = 0.0;
  } // end for (ii = 0; ii < inum; ii++)

  double rmj, rIj;
  for (ii = 0; ii < inum; ii++) {
    for (int kf = 0; kf <= ns[ii]; kf++) {
      j = ls[ii][kf];
      rmj = rmass[j];
      radj = radius[j];
      rIj = 0.5*rmj*radj*radj;
      mvx[j] += rmj*vs[ii][kf][0];
      mvy[j] += rmj*vs[ii][kf][1];
      m0t[j] += rmj;
      mvz[j] += rIj*vs[ii][kf][2];
      mIt[j] += rIj;
    } // end for (int kf = 1; kf <= ns[ii]; kf++) {
  } // end for (ii = 0; ii < inum; ii++)

    double vnmag, dvmag, dvfac, dvx, dvy, dvz;
    double elastic_wavespeed = sqrt(bulkModulus/rho0) + sqrt(shearModulus/rho0);
//  double bulk_CFL = fmax(1., bulkCFL);
    double rcfac = 200.;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    dvx = mvx[i]/m0t[i]; dvy = mvy[i]/m0t[i]; dvz = mvz[i]/mIt[i];
    radi = radius[i];
    dvmag = dvx*dvx + dvy*dvy + (radi*dvz)*(radi*dvz);
    double cfc = sqrt(dvmag)*(dtv/radi);  // convective cfl
        double bulk_CFL = fmax(1., elastic_wavespeed*dtv/radi);
        dvfac = fmax(1., fmax(1., rcfac/(bulk_CFL*bulk_CFL))*cfc);
    if(bulk_CFL >= 1.) {dvx /= dvfac; dvy /= dvfac; dvz /= dvfac;}

//  vn[i][0] += dvx; vn[i][1] += dvy; vn[i][2] += dvz;
    vn[i][0] = v[i][0] + dvx;
    vn[i][1] = v[i][1] + dvy;
    vn[i][2] = omega[i][2] + dvz;
  } // end for (ii = 0; ii < inum; ii++)

  delete[] ns, ls, vs;

} // end PairGranRate::compute_rate_implicit
/* ---------------------------------------------------------------------- */


void PairGranRate::load_new_forces()
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
  double *ice_area = atom->ice_area;
  double *mean_thickness = atom->mean_thickness;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int bondFlagIn; // incoming bondFlag
  int Fplus;      // outgoing bondFlag

  double delx, dely, rsq, radsum;
  double radi, radj, r, rinv, nx, ny, tx, ty;
  double vrx, vry, vnnr, wrz, vttr;
  double hAVGi, hAVGj, hDeltal;
  double fx, fy, fz;
  double Pplus, Splus, Sxplus, Syplus; // outgoing Pressure, ShearStresses
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
         *history[1]: bondShear
         *history[2]: bondFlag [0=no bond, 1=bonded]]
      */

      delx = x[j][0] - x[i][0]; // outward normal relative to [i]
      dely = x[j][1] - x[i][1];
      rsq = delx*delx + dely*dely;
      radsum = 1.10*(radi + radj);

      bondFlagIn = int(history[2]);
      Fplus = bondFlagIn;

      if(!bondFlagIn){
        if (rsq >= radsum*radsum){ // no contact
          firsttouch[i][jj] = 0;
          Fplus = 0;
          Pplus = 0.;
          Splus = 0.;
          fx = fy = fz = 0.;
        } else { // in first contact
          firsttouch[i][jj] = 1;
          Fplus = 1;
        } // end if(rsq >= radsum*radsum
        history[0] = history[1] = 0.;
      } // end if(!bondFlagIn)

//    if (Fplus){
        r = sqrt(rsq);
        nx = delx/r;
        ny = dely/r;
        tx = -ny;
        ty =  nx;
        rinv = 1.0/r;

        // relative translational velocity vector
        vrx = vn[j][0] - vn[i][0];
        vry = vn[j][1] - vn[i][1];

        // relative rotational velocity
        double radij = 2*(radi * radj)/(radi + radj); // Harmonic mean radius
        wrz = -radij*(vn[i][2] + vn[j][2]);

        // magnitude of the normal relative velocity
        vnnr = vrx*nx + vry*ny;

        // magnitude of the relative tangential velocity
        vttr = vrx*tx + vry*ty + wrz;

      if (Fplus){
        double dLogV = dtv*vnnr*rinv;
        double dLogS = dtv*vttr*rinv;
        Pplus = history[0]  + bulkModulus*dLogV; // full step stress
        Splus = history[1] + shearModulus*dLogS;
        omega[i][0] += dLogV;
        omega[i][1] += dLogS;
        omega[j][0] += dLogV;
        omega[j][1] += dLogS;
      } else { // add neighborwise smoothing stress 
          Pplus =  bulkModulus*dtv*vnnr*rinv/4.;
          Splus = shearModulus*dtv*vttr*rinv/4.;
      }

      Sxplus = Splus*tx;
      Syplus = Splus*ty;

       double CFLfac = 1. + pow(fmax(1.,bulkCFL),2.);
       // test for tensile fracture, compressive flowstress, shear flowstress
//     if (jtype != 2) { // do not limit stresses on the coastline (~noslip boundary condition)
          if (Pplus >= CFLfac*tensileFractureStress) { // tensile fracture
            Pplus = 0.;
            Splus = 0.; // maybe
            Fplus = 0;
            v[i][2] += 1.;
            v[j][2] += 1.;
          } else if (Pplus < CFLfac*compressiveYieldStress) { // perfectly plastic compressive flowstress
            Pplus = CFLfac*compressiveYieldStress;
            omega[i][0] += 1.;
            omega[j][0] += 1.;
          }
          SYield = sqrt(Sxplus*Sxplus + Syplus*Syplus)/(CFLfac*shearYieldStress);
          if (SYield > 1.0) { // perfectly plastic shear flowstress
            Splus /= SYield;
            omega[i][1] += 1.;
            omega[j][1] += 1.;
          }
//      } // end if (jtype != 2)

        // update history
        if(historyupdate){
          history[0] = Pplus;
          history[1] = Splus;
          history[2] = double(Fplus);
        }

        Sxplus = Splus*tx;
        Syplus = Splus*ty;

        hAVGi = mean_thickness[i];
        hAVGj = mean_thickness[j];
        hDeltal = 2.*MY_PI3*(radi*hAVGi * radj*hAVGj)
                           /(radi*hAVGi + radj*hAVGj); // contact Area

        fx = (Pplus*nx + Sxplus)*hDeltal;
        fy = (Pplus*ny + Syplus)*hDeltal;
        fz = -radij*Splus*hDeltal;

        f[i][0] += fx;
        f[i][1] += fy;
        torque[i][2] += fz;
        
        if (force->newton_pair || j < atom->nlocal){
          f[j][0] -= fx;
          f[j][1] -= fy;
          torque[j][2] += fz;
        }

      if (evflag) ev_tally_xyz(i,j,atom->nlocal, force->newton_pair,
              0.0,0.0,fx,fy,0,x[i][0]-x[j][0],x[i][1]-x[j][1],0);

    } // end for (ii = 0; ii < inum; ii++)
  } // end for (ii = 0; ii < inum; ii++)
} // end PairGranRate::load_new_forces()

/* ----------------------------------------------------------------------
   global settings
   ------------------------------------------------------------------------- */

void PairGranRate::settings(int narg, char **arg)
{
  if (narg != 7) error->all(FLERR,"Illegal pair_style command");
             bulkModulus = utils::numeric(FLERR, arg[0],false,lmp);
            shearModulus = utils::numeric(FLERR, arg[1],false,lmp);
        shearYieldStress = utils::numeric(FLERR, arg[2],false,lmp);
  compressiveYieldStress = utils::numeric(FLERR, arg[3],false,lmp);
   tensileFractureStress = utils::numeric(FLERR, arg[4],false,lmp);
     timeIntegrationFlag = utils::numeric(FLERR, arg[5],false,lmp);
                 bulkCFL = utils::numeric(FLERR, arg[6],false,lmp);
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
