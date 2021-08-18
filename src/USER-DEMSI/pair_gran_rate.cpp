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

  if (int(UseTVTransport) == 1) compute_TV_Transport(evflag);

  // insert a thread (MPI?) BARRIER here ???  "fence" ???

  if       (int(timeIntegrationFlag) == 0) { compute_rate_exverlet(evflag);

  } else if(int(timeIntegrationFlag) == 1) { compute_rate_explicit(evflag);

  } else if(int(timeIntegrationFlag) == 2) { compute_rate_implicit(evflag);

  } else {

    error->all(FLERR,"int(timeIntegrationFlag) must be (0,1,2)");

  }

  // insert a thread (MPI?) BARRIER here ???  "fence" ???

  load_new_forces(evflag);

  if (vflag_fdotr) virial_fdotr_compute();

} // end PairGranRate::compute
/* ---------------------------------------------------------------------- */

void PairGranRate::compute_TV_Transport(int evflag) {
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
//double dtf = 0.5 * update->dt; // * force->ftm2v;

  double elastic_wavespeed = sqrt(bulkModulus/rho0) + sqrt(shearModulus/rho0);

  // get the averaged v, omega, in the neighborhood of each element
  double mvx[inum], mvy[inum], mvz[inum], m0t[inum], mIt[inum];
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    mvx[i] = mvy[i] = mvz[i] = m0t[i] = mIt[i] = 0.0;
  } // end for (ii = 0; ii < inum; ii++)

  double rmj, rIj;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    if (itype != 1) continue;

    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      jtype = atom->type[j];
//    if (jtype != 1) continue;
//    j &= NEIGHMASK;
       rmj = rmass[j];
      radj = radius[j];
       rIj = 0.5*rmj*radj*radj;
      mvx[i] += rmj*v[j][0];
      mvy[i] += rmj*v[j][1];
      m0t[i] += rmj;
      mvz[i] += rIj*omega[j][2];
      mIt[i] += rIj;
    } // end for (int kf = 1; kf <= ns[ii]; kf++) {
  } // end for (ii = 0; ii < inum; ii++)

  // add the Tensor Viscosity acceleration
  double dvx, dvy, dvz, dvsqr, dMach;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    if (itype != 1) continue;

    if (m0t[i] > 0.) {
      double rmt = 1./m0t[i];
      dvx = mvx[i]*rmt - v[i][0];
      dvy = mvy[i]*rmt - v[i][1];
      dvz = mvz[i]/mIt[i] - omega[i][2];
      radi = radius[i];
      dvsqr = dvx*dvx + dvy*dvy + (radi*dvz)*(radi*dvz); // |dv|^2
      dMach = 0.25*sqrt(dvsqr)*dtv/radi;
    } else {
      dvx = dvy = dvz = dMach = 0.;
    }
    torque[i][0] = dMach; // diagnostic
    if ( bulkCFL >= 1. ) {
          dMach = fmin(1.,dMach);
          v[i][0] += dMach*dvx;
          v[i][1] += dMach*dvy;
      omega[i][2] += dMach*dvz;
    }
  } // end for (ii = 0; ii < inum; ii++)

} // end PairGranRate::compute_TV_Transport
/* ---------------------------------------------------------------------- */

void PairGranRate::compute_rate_exverlet(int evflag) {
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

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    if (itype != 1) continue;

    vn[i][0] =     v[i][0];
    vn[i][1] =     v[i][1];
    vn[i][2] = omega[i][2];
  } // end for (ii = 0; ii < inum; ii++)

} // end PairGranRate::compute_rate_verlet
/* ---------------------------------------------------------------------- */

void PairGranRate::compute_rate_explicit(int evflag)
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
  if (update->setupflag) return;

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
    if (itype != 1) continue;

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
         *history[0]: contactPressure
         *history[1]: contactShear
         *history[2]: contactFlag [0=no contact, 1=in contact]]
      */

      delx = x[j][0] - x[i][0]; // outward normal relative to [i]
      dely = x[j][1] - x[i][1];
      rsq = delx*delx + dely*dely;
      radsum = 1.10*(radi + radj);
//    radsum = 1.00*(radi + radj);

        r = sqrt(rsq);
        rinv = 1./r;
        nx = delx*rinv;
        ny = dely*rinv;
        tx = -ny;
        ty =  nx;

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

      if ( Fplus != 1 ) {
        Pplus = Splus = 0.;
      } else {
        double dLogV = dtf*vnnr*rinv;
        double dLogS = dtf*vttr*rinv;
        Pplus =  bulkModulus*dLogV; // half step stress increment
        Splus = shearModulus*dLogS;
      }

       // test for tensile fracture, compressive flowstress, shear flowstress
          if (Pplus >= tensileFractureStress) { // perfectly plastic tensile flowstress with fracture
            Pplus = tensileFractureStress;
            if (log(r/radsum) > ultimateTensileStrain) { // fracture
              Pplus = 0.; Fplus = 0;
            }
          } else if (Pplus < compressiveYieldStress) { // perfectly plastic compressive flowstress
            Pplus = compressiveYieldStress;
          }
          Sxplus = Splus*tx;
          Syplus = Splus*ty;
          SYield = sqrt(Sxplus*Sxplus + Syplus*Syplus)/shearYieldStress;
          if (SYield > 1.0) { // perfectly plastic shear flowstress
            Splus /= SYield;
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

    } // end for (jj = 0; jj < jnum; jj++)
  } // end for (ii = 0; ii < inum; ii++)

  // full step with the half step stress increment
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    if (itype != 1) continue;

    double dtm = dtv/rmass[i];
    radi = radius[i];
    double dvx = dtm*f[i][0];
    double dvy = dtm*f[i][1];
    double dvz = dtm*torque[i][2]/(0.5*radi*radi);

    vn[i][0] =     v[i][0] + dvx;
    vn[i][1] =     v[i][1] + dvy;
    vn[i][2] = omega[i][2] + dvz;

  } // end for (ii = 0; ii < inum; ii++)

} // end PairGranRate::compute_rate_explicit
/* ---------------------------------------------------------------------- */

void PairGranRate::compute_rate_implicit(int evflag)
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
  if (update->setupflag) return;

  double dtv = update->dt;
  double dtf = 0.5*dtv;
  double mvx[inum], mvy[inum], mvz[inum], m0t[inum], mIt[inum];
  double rmj, rIj;

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
  const int ndim = 2;
/* work space: uses indices starting with 1 */
  const int wdim = (1 + nmax)*ndim;
  double A[wdim][wdim];  // upper echelon matrix of coefficients
  double P[wdim][wdim];  // the full matrix of pivots
  double b[wdim];        // the right-side vector, and solution vector
//auto A = new double[wdim][wdim];
//auto P = new double[wdim][wdim];
//auto b = new double[wdim];

/* load the lists ns, ls, vs */
  int nsp;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    if (itype != 1) continue;

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
//    j &= NEIGHMASK; // is this needed here ???
      history = &allhistory[size_history*jj];
      Fplus = int(history[2]);
//    if (Fplus == 1) { // the neighbor is "in contact"
        nsp = ns[ii] + 1;
        if(nsp > nmax-1) break; // should never happen, but we need to check...
        ns[ii] = nsp;
        ls[ii][nsp] = j;
        vs[ii][nsp][0] = v[j][0];
        vs[ii][nsp][1] = v[j][1];
        vs[ii][nsp][2] = omega[j][2];
//    } // end if Fplus == 1
    } // end for (jj = 0; jj < jnum; jj++)
  } // end for (ii = 0; ii < inum; ii++)

  // set up and solve for the implicit acceleration
  for (ii = 0; ii < inum; ii++) if (ns[ii] > 0) {
    i = ilist[ii];
    itype = atom->type[i];
    if (itype != 1) continue;

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
        rinv = 1./r;
        nx = delx*rinv;
        ny = dely*rinv;
        tx = -ny;
        ty =  nx;

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

// 2d form works.
        double nn[2][2] = {{nx*nx,nx*ny},{ny*nx,ny*ny}};
        double tt[2][2] = {{tx*tx,tx*ty},{ty*tx,ty*ty}};
/*
// 3d fully coupled form has faulty Delta[omega].  There may be a sign error,
  // or the nonconservative system may not lend itself to the elimination method...

        double Rij = 2.*(radi*radj)/(radi+radj);
        double  GK = 2.*Rij/(radi*radi);
        double  GL = 2.*Rij/(radj*radj);
        double  nn[3][3] = {{nx*nx,nx*ny,     0.},{ny*nx,ny*ny,     0.},{    0.,    0.,     0.}};
        double ttK[3][3] = {{tx*tx,tx*ty, Rij*tx},{ty*tx,ty*ty, Rij*ty},{-GK*tx,-GK*ty,-GK*Rij}};
        double tmK[3][3] = {{tx*tx,tx*ty,-Rij*tx},{ty*tx,ty*ty,-Rij*ty},{-GK*tx,-GK*ty, GK*Rij}};
        double ttL[3][3] = {{tx*tx,tx*ty, Rij*tx},{ty*tx,ty*ty, Rij*ty},{-GL*tx,-GL*ty,-GL*Rij}};
        double tmL[3][3] = {{tx*tx,tx*ty,-Rij*tx},{ty*tx,ty*ty,-Rij*ty},{-GL*tx,-GL*ty, GL*Rij}};
*/
        for (int jd = 1; jd <= ndim; jd++){
          int neq_k = mf*(jd-1) + kf;
          int neq_l = mf*(jd-1) + lf;
          for (int id = 1; id <= ndim; id++){
            int ind_kf = mf*(id-1) + kf;
            int ind_lf = mf*(id-1) + lf;

            // 2d form
            A[neq_k][ind_kf] = A[neq_k][ind_kf] + (kappaK*nn[id-1][jd-1] + gammaK*tt[id-1][jd-1]);
            A[neq_k][ind_lf] = A[neq_k][ind_lf] - (kappaK*nn[id-1][jd-1] + gammaK*tt[id-1][jd-1]);
            A[neq_l][ind_lf] = A[neq_l][ind_lf] + (kappaL*nn[id-1][jd-1] + gammaL*tt[id-1][jd-1]);
            A[neq_l][ind_kf] = A[neq_l][ind_kf] - (kappaL*nn[id-1][jd-1] + gammaL*tt[id-1][jd-1]);
/*
            // 3d fully coupled form
            A[neq_k][ind_kf] = A[neq_k][ind_kf] + (kappaK*nn[id-1][jd-1] + gammaK*ttK[id-1][jd-1]);
            A[neq_k][ind_lf] = A[neq_k][ind_lf] - (kappaK*nn[id-1][jd-1] + gammaK*tmK[id-1][jd-1]);
            A[neq_l][ind_lf] = A[neq_l][ind_lf] + (kappaL*nn[id-1][jd-1] + gammaL*ttL[id-1][jd-1]);
            A[neq_l][ind_kf] = A[neq_l][ind_kf] - (kappaL*nn[id-1][jd-1] + gammaL*tmL[id-1][jd-1]);
*/
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
//        vs[ii][kf-1][jd-1] = b[neq_k] - vn[i][jd-1];
          vs[ii][kf-1][jd-1] = b[neq_k];
      }
    }
  } // end for (ii = 0; ii < inum; ii++) if(ns[ii] > 0) {

// here we need to communicate ns,ls,vs ... otherwise, just load vn ???

  // get the averaged time(n+1) v; and load it into vn
  
//double mvx[inum], mvy[inum], mvz[inum], m0t[inum], mIt[inum];
  for (ii = 0; ii < inum; ii++) {
    i = ls[ii][0];
    mvx[i] = mvy[i] = mvz[i] = m0t[i] = mIt[i] = 0.0;
  } // end for (ii = 0; ii < inum; ii++)

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    if (itype != 1) continue;
    for (int kf = 0; kf <= ns[ii]; kf++) {
      j = ls[ii][kf];
      rmj = rmass[j];
      radj = radius[j];
      rIj = 0.5*rmj*radj*radj;
      mvx[j] += rmj*vs[ii][kf][0];
      mvy[j] += rmj*vs[ii][kf][1];
      m0t[j] += rmj;
      mvz[j] += rIj*omega[j][2]; // 2d form
//    mvz[j] += rIj*vs[ii][kf][2]; // 3d form
      mIt[j] += rIj;
    } // end for (int kf = 1; kf <= ns[ii]; kf++) {
  } // end for (ii = 0; ii < inum; ii++)

  // full step with the half step stress increment
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atom->type[i];
    if (itype != 1) continue;

    double dvx = mvx[i]/m0t[i];
    double dvy = mvy[i]/m0t[i];
    double dvz = mvz[i]/mIt[i];

//  if (dvx != dvx) continue; // NAN trap for debugging

    vn[i][0] = dvx;
    vn[i][1] = dvy;
    vn[i][2] = dvz;
  } // end for (ii = 0; ii < inum; ii++)

  delete[] ns, ls, vs;

} // end PairGranRate::compute_rate_implicit
/* ---------------------------------------------------------------------- */

void PairGranRate::load_new_forces(int evflag)
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
// in the RATE_FORM case we must do the history update at setup in order
// to properly initialize the history variables!!!
//if (update->setupflag) historyupdate = 0;

  double dtv = update->dt;
//double dtf = 0.5 * update->dt; // * force->ftm2v;

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
         *history[0]: contactPressure
         *history[1]: contactShear
         *history[2]: contactFlag [0=no contact, 1=in contact]]
      */

      delx = x[j][0] - x[i][0]; // outward normal relative to [i]
      dely = x[j][1] - x[i][1];
      rsq = delx*delx + dely*dely;
      radsum = 1.10*(radi + radj);
//    radsum = 1.00*(radi + radj);

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
        rinv = 1./r;
        nx = delx*rinv;
        ny = dely*rinv;
        tx = -ny;
        ty =  nx;

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
//      if ( vn[i][0] == vn[i][0] ) { // NAN trap for debugging
          omega[i][0] += dLogV/2.;
          omega[i][1] += dLogS/2.;
          if (jtype == 1) {
            omega[j][0] += dLogV/2.;
            omega[j][1] += dLogS/2.;
          }
//      }
      } else { // add neighborwise smoothing stress, perhaps...
        Pplus = 0.;
        Splus = 0.;
      }

       // test for tensile fracture, compressive flowstress, shear flowstress
//     if (jtype == 1) { // do not limit stresses on the coastline (~noslip boundary condition)
          if (Pplus >= tensileFractureStress) { // perfectly plastic tensile flowstress with fracture
            Pplus = tensileFractureStress;
//          double tensile_strain = log(r/radsum);
            if (log(r/radsum) > ultimateTensileStrain) { // fracture
              Pplus = 0.; Fplus = 0;
            }
//          if (itype == 1 and jtype == 1) {v[i][2] += 1.; v[j][2] += 1.;}
          } else if (Pplus < compressiveYieldStress) { // perfectly plastic compressive flowstress
            Pplus = compressiveYieldStress;
          }
          Sxplus = Splus*tx;
          Syplus = Splus*ty;
          SYield = sqrt(Sxplus*Sxplus + Syplus*Syplus)/shearYieldStress;
          if (SYield > 1.0) { // perfectly plastic shear flowstress
            Splus /= SYield;
          }
//      } // end if (jtype == 1) !! causes MPI fault

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
} // end PairGranRate::load_new_forces()

/* ----------------------------------------------------------------------
   global settings
   ------------------------------------------------------------------------- */

void PairGranRate::settings(int narg, char **arg)
{
  if (narg != 8) error->all(FLERR,"Illegal pair_style command");
             bulkModulus = utils::numeric(FLERR, arg[0],false,lmp);
            shearModulus = utils::numeric(FLERR, arg[1],false,lmp);
        shearYieldStress = utils::numeric(FLERR, arg[2],false,lmp);
  compressiveYieldStress = utils::numeric(FLERR, arg[3],false,lmp);
   tensileFractureStress = utils::numeric(FLERR, arg[4],false,lmp);
     timeIntegrationFlag = utils::numeric(FLERR, arg[5],false,lmp);
                 bulkCFL = utils::numeric(FLERR, arg[6],false,lmp);
          UseTVTransport = utils::numeric(FLERR, arg[7],false,lmp);

   ultimateTensileStrain = 2.*tensileFractureStress/bulkModulus;
// ultimateTensileStrain = 1.;
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
