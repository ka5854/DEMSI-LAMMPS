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

#include "fix_nve_sphere_demsi_kokkos.h"
#include "atom_masks.h"
#include "atom_kokkos.h"
#include "error.h"

using namespace LAMMPS_NS;

enum{NONE,DIPOLE};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixNVESphereDemsiKokkos<DeviceType>::FixNVESphereDemsiKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixNVESphereDemsi(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *)atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = X_MASK | V_MASK| F_MASK | OMEGA_MASK | TORQUE_MASK |
   RMASS_MASK | RADIUS_MASK | MASK_MASK | THICKNESS_MASK | VN_MASK; // adding vn

  datamask_modify = X_MASK | V_MASK | OMEGA_MASK |
   THICKNESS_MASK | VN_MASK; // adding vn
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVESphereDemsiKokkos<DeviceType>::cleanup_copy()
{
  id = style = nullptr;
  vatom = nullptr;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVESphereDemsiKokkos<DeviceType>::init()
{
  FixNVESphereDemsi::init();

//  if (extra == DIPOLE) {
//    error->all(FLERR,"Fix nve/sphere/demsi/kk doesn't yet support dipole");
//  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVESphereDemsiKokkos<DeviceType>::initial_integrate(int vflag)
{
  atomKK->sync(execution_space,datamask_read);
  atomKK->modified(execution_space,datamask_modify);

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  omega = atomKK->k_omega.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  torque = atomKK->k_torque.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  radius = atomKK->k_radius.view<DeviceType>();
  orientation = atomKK->k_orientation.view<DeviceType>();
  momentOfInertia = atomKK->k_momentOfInertia.view<DeviceType>();
  ice_area = atomKK->k_ice_area.view<DeviceType>();
  coriolis = atomKK->k_coriolis.view<DeviceType>();
  ocean_vel = atomKK->k_ocean_vel.view<DeviceType>();
  bvector = atomKK->k_bvector.view<DeviceType>();
  forcing = atomKK->k_forcing.view<DeviceType>();
  mean_thickness = atomKK->k_mean_thickness.view<DeviceType>();
  vn = atomKK->k_vn.view<DeviceType>(); // adding vn

  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  FixNVESphereDemsiKokkosInitialIntegrateFunctor<DeviceType> f(this); 
  Kokkos::parallel_for(nlocal,f);

//debug
//atomKK->sync(Host,ALL_MASK);
//atomKK->modified(Host,ALL_MASK);

}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVESphereDemsiKokkos<DeviceType>::initial_integrate_item(const int i) const
{

  if (mask(i) & groupbit) {

    // half step acceleration with time(n) forces
    double dtm = dtf/rmass(i);
    double dvx = dtm*f(i,0);
    double dvy = dtm*f(i,1);
    double dvz = dtf*torque(i,2)/momentOfInertia(i);
    double un1 =     v(i,0);
    double vn1 =     v(i,1);
    double wn1 = omega(i,2);

    // add the forcing
    double u1 = un1 + dvx + dtm*forcing(i,0);
    double v1 = vn1 + dvy + dtm*forcing(i,1);

    // 3-field momentum exchange: field 1-ice, 2-atmosphere, 3-ocean
    double ice_Area = ice_area(i);
    double ice_Volume = fmax(1.e-99, ice_Area*mean_thickness(i));
    double ice_Density = rmass(i)/ice_Volume;
    double u2 = bvector(i,0);
    double v2 = bvector(i,1);
    double u3 = ocean_vel(i,0);
    double v3 = ocean_vel(i,1);
    // w12 = ice-atmos relative speed
    double w12 = sqrt(pow(u2-u1,2) + pow(v2-v1,2));
    // w13 = ice-ocean relative speed
    double w13 = sqrt(pow(u3-u1,2) + pow(v3-v1,2));
    double D12 = atmos_Density*atmos_Drag*w12*ice_Area;
    double D13 = ocean_Density*ocean_Drag*w13*ice_Area;
    double a = D12*dtf/(ice_Density*ice_Volume);
    double b = D13*dtf/(ice_Density*ice_Volume);

    // fixed atmos vel, fixed ocean vel:
    double up1 = (u1 + a*u2 + b*u3)/(1. + a + b);
    double vp1 = (v1 + a*v2 + b*v3)/(1. + a + b);

    // half step coriolis acceleration
    double cdt = coriolis(i)*dtf;
        v(i,0) = (up1 + cdt*vp1)/(1. + cdt*cdt);
        v(i,1) = (vp1 - cdt*up1)/(1. + cdt*cdt);
/*
    // rotational drag: zero ocean curl, zero atmos curl
    double w1 = wn1 + dvz;
    double w2 = 0.;  // placeholder for curl(u2)
    double w3 = 0.;  // placeholder for curl(u3)
    w12 = fabs(w2 - w1);
    w13 = fabs(w3 - w1);
    double radi = radius(i);
    D12 = atmos_Density*atmos_Drag*w12*ice_Area*radi;
    D13 = ocean_Density*ocean_Drag*w13*ice_Area*radi;
    a = D12*0.5*dtf/(ice_Density*ice_Volume);
    b = D13*0.5*dtf/(ice_Density*ice_Volume);
//  omega(i,2) = (w1 + a*w2 + b*w3)/(1. + a + b);
    dvz = (w1 + a*w2 + b*w3)/(1. + a + b) - wn1;
*/
    // semi-implicit rotational acceleration
    if (wn1 + 0.5*dvz != 0.) {
      double dom = dvz/(wn1 + 0.5*dvz); // ~dLogOmega
      omega(i,2) *= (1. + fmax(0.,dom))/(1. - fmin(0.,dom));
    } else {
      omega(i,2) += dvz;
    }

    x(i,0) += dtv * v(i,0);  // full step with explicit vel
    x(i,1) += dtv * v(i,1);
    orientation(i) += dtv * omega(i,2);

    // put v, omega into vn for modification in PairGranRate::compute
    vn(i,0) =     v(i,0);
    vn(i,1) =     v(i,1);
    vn(i,2) = omega(i,2);

  } // end if (mask(i) & groupbit)

  // zero out diagnostics on nonmobile elements
  if(!(mask(i)&groupbit)) {
            vn(i,0) =     vn(i,1) =     vn(i,2) = 0.;
        torque(i,0) = torque(i,1) = torque(i,2) = 0.;
         omega(i,0) =  omega(i,1) =  omega(i,2) = 0.;
  }

} // end initial_integrate_item


/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVESphereDemsiKokkos<DeviceType>::final_integrate()
{
  atomKK->sync(execution_space,datamask_read);
  atomKK->modified(execution_space,datamask_modify);

  v = atomKK->k_v.view<DeviceType>();
  omega = atomKK->k_omega.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  torque = atomKK->k_torque.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  orientation = atomKK->k_orientation.view<DeviceType>();
  momentOfInertia = atomKK->k_momentOfInertia.view<DeviceType>();
  radius = atomKK->k_radius.view<DeviceType>();
  ice_area = atomKK->k_ice_area.view<DeviceType>();
  coriolis = atomKK->k_coriolis.view<DeviceType>();
  ocean_vel = atomKK->k_ocean_vel.view<DeviceType>();
  bvector = atomKK->k_bvector.view<DeviceType>();
  forcing = atomKK->k_forcing.view<DeviceType>();
  mean_thickness = atomKK->k_mean_thickness.view<DeviceType>();
  vn = atomKK->k_vn.view<DeviceType>(); // adding vn

  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  FixNVESphereDemsiKokkosFinalIntegrateFunctor<DeviceType> f(this);
  Kokkos::parallel_for(nlocal,f);

//debug
//atomKK->sync(Host,datamask_read);
//atomKK->modified(Host,datamask_modify);
//atomKK->sync(Host,ALL_MASK);
//atomKK->modified(Host,ALL_MASK);

}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVESphereDemsiKokkos<DeviceType>::final_integrate_item(const int i) const
{

  if (mask(i) & groupbit) {

    // get v, omega which were modified in PairGranRate::compute
        v(i,0) = vn(i,0);
        v(i,1) = vn(i,1);
    omega(i,2) = vn(i,2);

    // half step acceleration with time(n+1) forces
    double dtm = dtf/rmass(i);
    double dvx = dtm*f(i,0);
    double dvy = dtm*f(i,1);
    double dvz = dtf*torque(i,2)/momentOfInertia(i);
    double un1 =     v(i,0);
    double vn1 =     v(i,1);
    double wn1 = omega(i,2);

    // add the forcing
    double u1 = un1 + dvx + dtm*forcing(i,0);
    double v1 = vn1 + dvy + dtm*forcing(i,1);

    // 3-field momentum exchange: field 1-ice, 2-atmosphere, 3-ocean
    double ice_Area = ice_area(i);
    double ice_Volume = fmax(1.e-99, ice_Area*mean_thickness(i));
    double ice_Density = rmass(i)/ice_Volume;
    double u2 = bvector(i,0);
    double v2 = bvector(i,1);
    double u3 = ocean_vel(i,0);
    double v3 = ocean_vel(i,1);
    // w12 = ice-atmos relative speed
    double w12 = sqrt(pow(u2-u1,2) + pow(v2-v1,2));
    // w13 = ice-ocean relative speed
    double w13 = sqrt(pow(u3-u1,2) + pow(v3-v1,2));

    double D12 = atmos_Density*atmos_Drag*w12*ice_Area;
    double D13 = ocean_Density*ocean_Drag*w13*ice_Area;
    double a = D12*dtf/(ice_Density*ice_Volume);
    double b = D13*dtf/(ice_Density*ice_Volume);

    // fixed atmos vel, fixed ocean vel:
    double up1 = (u1 + a*u2 + b*u3)/(1. + a + b);
    double vp1 = (v1 + a*v2 + b*v3)/(1. + a + b);

    // half step coriolis acceleration
    double cdt = coriolis(i)*dtf;
        v(i,0) = (up1 + cdt*vp1)/(1. + cdt*cdt);
        v(i,1) = (vp1 - cdt*up1)/(1. + cdt*cdt);
/*
    // rotational drag: zero ocean curl, zero atmos curl
    double w1 = wn1 + dvz;
    double w2 = 0.;  // placeholder for curl(u2)
    double w3 = 0.;  // placeholder for curl(u3)
    w12 = fabs(w2 - w1);
    w13 = fabs(w3 - w1);
    double radi = radius(i);
    D12 = atmos_Density*atmos_Drag*w12*ice_Area*radi;
    D13 = ocean_Density*ocean_Drag*w13*ice_Area*radi;
    a = D12*0.5*dtf/(ice_Density*ice_Volume);
    b = D13*0.5*dtf/(ice_Density*ice_Volume);
//  omega(i,2) = (w1 + a*w2 + b*w3)/(1. + a + b);
    dvz = (w1 + a*w2 + b*w3)/(1. + a + b) - wn1;
*/
    // semi-implicit rotational acceleration
    if (wn1 + 0.5*dvz != 0.) {
      double dom = dvz/(wn1 + 0.5*dvz); // ~dLogOmega
      omega(i,2) *= (1. + fmax(0.,dom))/(1. - fmin(0.,dom));
    } else {
      omega(i,2) += dvz;
    }

  } // end if (mask(i) & groupbit)

  // zero out diagnostics on nonmoving elements
  if(!(mask(i)&groupbit)) {
            vn(i,0) =     vn(i,1) =     vn(i,2) = 0.;
        torque(i,0) = torque(i,1) = torque(i,2) = 0.;
         omega(i,0) =  omega(i,1) =  omega(i,2) = 0.;
  }

  // diagnostics for use in DEMSI::ParticlesWrite::write
  const double PIE = 3.14159265358979323846; // pi
  const double rho = 900.;
  const double rhoPIE  = rho*PIE; // pi
  double radi = radius(i);
  
  vn(i,0) = rmass(i)/(rhoPIE*radi*radi); // dLogV
  vn(i,1) = mean_thickness(i); // dLogS
  vn(i,2) = torque(i,0); // dMach

} // end final_integrate_item

namespace LAMMPS_NS {
template class FixNVESphereDemsiKokkos<LMPDeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
template class FixNVESphereDemsiKokkos<LMPHostType>;
#endif
}
