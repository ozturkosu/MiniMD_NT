/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"

#include "neighbor.h"
#include "openmp.h"

#define FACTOR 0.999
#define SMALL 1.0e-6

#ifdef  __INTEL_COMPILER
#include <ia32intrin.h>
#include <xmmintrin.h>
#include <zmmintrin.h>
#endif

Neighbor::Neighbor()
{
  ncalls = 0;
  max_totalneigh = 0;
  numneigh = NULL;
  neighbors = NULL;
  // maxneighs = 100;  //hma
  maxneighs = 3000; //hma
  nmax = 0;
  bincount = NULL;
  bins = NULL;
  atoms_per_bin = 8;
  stencil = NULL;
  threads = NULL;
  halfneigh = 0;
  ghost_newton = 1;

  /* hma */
  totalneighs = 0;
  npair_stencil = 0;
  pair_stencil = NULL;
  locbin = NULL;
  num_cbins = NULL;
  cbin_list = NULL;
 /* mathi */
  numcbin = 0;		//-mathi
  cbin = NULL;
  tempF = NULL;
  tempnumneigh = NULL;
  atoms_per_thread = 0;
}

Neighbor::~Neighbor()
{
#ifdef ALIGNMALLOC
  if(numneigh) _mm_free(numneigh);

  if(neighbors) _mm_free(neighbors);

  if(bincount) _mm_free(bincount);

  if(bins) _mm_free(bins);
  /* mathi*/
  if(cbin) _mm_free(cbin);	//-mathi
  if (tempF) _mm_free(tempF);
#else 
  if(numneigh) free(numneigh);

  if(neighbors) free(neighbors);

  if(bincount) free(bincount);

  if(bins) free(bins);
  /* math*/
  if(cbin) free(cbin);
  if (tempF) free(tempF);
#endif  
}

/* binned neighbor list construction with full Newton's 3rd law
   every pair stored exactly once by some processor
   each owned atom i checks its own bin and other bins in Newton stencil */
#ifdef KNC_NEIGHBUILD_INTRINSIC_SWGS 
//optimised version of compute with INTRINSIC
//Ashish Jha, ashish.jha@intel.com, Intel Corporation
#endif

void Neighbor::build(Atom &atom)
{
  const int nlocal = atom.nlocal;
  const int nall = atom.nlocal + atom.nghost;
  FILE *f;
  
  if (ncalls == 1 && threads->mpi_me == 0)
    f = fopen("defneighs.out", "w");

#ifdef KNC_NEIGHBUILD_INTRINSIC_SWGS  
 __mmask8 k11 = _mm512_int2mask(0x11);
 __mmask8 k07 = _mm512_int2mask(0x07);
 __mmask8 k70 = _mm512_int2mask(0x70);
 __mmask8 k77 = _mm512_int2mask(0x77);
 __mmask8 kF0 = _mm512_int2mask(0xF0);
 __mmask8 kFF = _mm512_int2mask(0xFF);
#endif

  int omp_me = omp_get_thread_num();
  int num_omp_threads = threads->omp_num_threads;
  int master = -1;

  const MMD_float* x = &atom.x[0][0];
#ifdef KNC_NEIGHBUILD_INTRINSIC_SWGS  
  __m512d z_cutforcesq = _mm512_set_1to8_pd(cutneighsq);
#endif

  #pragma omp master
  {
    master = omp_me;
    ncalls++;
  }

  #pragma omp barrier
  /* bin local & ghost atoms */
  binatoms(atom);
  //atom.sort(this);
  
  count = 0;
  
  if ((hma == 1) ||(hma ==2 )||(hma ==3)) {
    int i, j, k, l, m, p;

    if (ncalls == 1) {
      int estimate = estimate_maxneighs_TP(atom);

      #pragma omp master
      {
      maxneighs = estimate;
      /* do the allocations */
    #ifdef ALIGNMALLOC
      if (numneigh) _mm_free(numneigh);
      if (neighbors) _mm_free(neighbors);
      numneigh = (int*) 
	_mm_malloc(mbins * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
      neighbors = (int*) 
	_mm_malloc(mbins*maxneighs * sizeof(int)*2 + ALIGNMALLOC, ALIGNMALLOC);
    #else
      if(numneigh) free(numneigh);
      if(neighbors) free(neighbors);
      numneigh = (int*) malloc(mbins * sizeof(int));
      neighbors = (int*) malloc(mbins * maxneighs * sizeof(int) * 2);
    #endif
      }
    }
  
    #pragma omp barrier
    /* compute neighbors */
    double wstart = omp_get_wtime();
    
#pragma omp for collapse(3) schedule(guided) 
    /* grab the central bin */
    for (k = locbinzlo; k <= locbinzhi; ++k) 
      for (j = locbinylo; j <= locbinyhi; ++j)
	for (i = locbinxlo; i <= locbinxhi; ++i) { 
	  const int center = k * mbiny * mbinx + j * mbinx + i + 1;
	  int* RESTRICT neighptr = &neighbors[center * maxneighs * 2];
	  int cntneighs = 0;

    	  /* neighbors within the central bin itself */
	  int* RESTRICT ctratoms = &bins[center * atoms_per_bin];
	  
	  for (l = 0; l < bincount[center]; ++l) {
	    const int a1 = ctratoms[l];
	    const MMD_float xtmp = x[a1 * PAD + 0];
	    const MMD_float ytmp = x[a1 * PAD + 1];
	    const MMD_float ztmp = x[a1 * PAD + 2];
	    
	    for (m = l+1; m < bincount[center]; ++m) {
	      const int a2 = ctratoms[m];
	      
	      if (a1 < nlocal || a2 < nlocal) {
		const MMD_float delx = xtmp - x[a2 * PAD + 0];
		const MMD_float dely = ytmp - x[a2 * PAD + 1];
		const MMD_float delz = ztmp - x[a2 * PAD + 2];
		const MMD_float rsq = delx * delx + dely * dely + delz * delz;
		
		if((rsq <= cutneighsq)) {
		  neighptr[cntneighs++] = a1;
		  neighptr[cntneighs++] = a2;
	     if (ncalls == 1 && threads->mpi_me == 0)
	     // printf("local %5d \n",a1);
	      printf("%5d  %5d  %5d\n", a1, a2, cntneighs); //MIN(a1,a2), MAX(a1,a2), rsq);
		}
	      }
	    }
	  }
	  
	  /* tower-plate neighbors */
	  for (p = 0; p < npair_stencil; p+=2) {
	    //binpair_cp[ pre-compute closest point of bins??
	    const int bin1 = center + pair_stencil[p];
	    const int bin2 = center + pair_stencil[p+1];
	    if (!locbin[bin1] && !locbin[bin2]) // if neither bin is local 
	      continue; 
	  
	    int* RESTRICT b1atoms = &bins[bin1 * atoms_per_bin];
	    int* RESTRICT b2atoms = &bins[bin2 * atoms_per_bin];
	    //printf("\t\tbin1 - bin2 pair: %d %d\n", bin1, bin2);   
	    
	    /* pick up an atom from bin1 */
	    for (l = 0; l < bincount[bin1]; ++l) {
	      const int a1 = b1atoms[l];
	      const MMD_float xtmp = x[a1 * PAD + 0];
	      const MMD_float ytmp = x[a1 * PAD + 1];
	      const MMD_float ztmp = x[a1 * PAD + 2];
	     // printf("\tatom %d\n", a1 );
	    
	      //if (distsqr_to_cp(nbrs_cp[p], x[a1]) <= cutneigh2) {
	      /* pick up another atom from the neighbor cell */
	      for (m = 0; m < bincount[bin2]; ++m) {
		const int a2 = b2atoms[m];
		
		if (a1 < nlocal || a2 < nlocal) {
		  const MMD_float delx = xtmp - x[a2 * PAD + 0];
		  const MMD_float dely = ytmp - x[a2 * PAD + 1];
		  const MMD_float delz = ztmp - x[a2 * PAD + 2];
		  const MMD_float rsq = delx*delx + dely*dely + delz*delz;
		
		  if((rsq <= cutneighsq)) {
		    neighptr[cntneighs++] = a1;
		    neighptr[cntneighs++] = a2;
	      if (ncalls == 1 && threads->mpi_me == 0)
	      printf("p %5d  %5d  %5d\n",a1, a2, cntneighs);// MIN(a1,a2), MAX(a1,a2), rsq);
		  }
		} 
	      } // for - bin2 atoms
	    } // for - bin1 atoms
	  } // for - bin pairs based on stencil
	  
	  numneigh[center] = cntneighs;
	}// for - local bins
  
    double welapsed = omp_get_wtime() - wstart;
    
    #pragma omp master
    {
      printf("neighbor generation took %.4f seconds\n", welapsed);
      printf("total number of neighbors is %d\n", totalneighs);
      //int cc =0;
      //for(int i=0; i<mbins; i++) cc+=numneigh[i];  
      //printf("cc : %5d\n", cc);
    //if (threads->mpi_me == 0) fclose(f);
     }

  }
else  if ((hma ==4)||(hma==5)){
	//vectorizable neighbor list
	
    int i, j, k, l, m, p;
    //int cn =0;
    if (ncalls == 1) {

      #pragma omp master
      {
       if(nall > nmax) 
          nmax = nall;
      /* do the allocations */
    #ifdef ALIGNMALLOC
	if (numneigh) _mm_free(numneigh);
      numneigh = (int*)
        _mm_malloc(nmax * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
	#else	
      if (numneigh) free(numneigh);
       numneigh = (int*)
        malloc(nmax * sizeof(int));
     #endif 
     for (int i =0; i< nmax; i++) numneigh[i] = 0; 
     }
      int estimate = estimate_maxneighs_TP(atom);
      #pragma omp master
      {
      maxneighs = estimate;
      /* do the allocations */
    #ifdef ALIGNMALLOC
      if (neighbors) _mm_free(neighbors);
      neighbors = (int*) 
	_mm_malloc(nmax *maxneighs * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
    #else
      if (neighbors) free(neighbors);
      neighbors = (int*) 
	malloc(nmax *maxneighs * sizeof(int));
    #endif
      }
    } 
    #pragma omp barrier
    /* compute neighbors */
    double wstart = omp_get_wtime();
   printf("nmax: %5d, maxneigh :%5d\n",nmax, maxneighs ); 
#pragma omp for collapse(3) schedule(guided) 
    /* grab the central bin */
    for (k = locbinzlo; k <= locbinzhi; ++k) 
      for (j = locbinylo; j <= locbinyhi; ++j)
	for (i = locbinxlo; i <= locbinxhi; ++i) { 
	  const int center = k * mbiny * mbinx + j * mbinx + i + 1;
//	  cn =0;
	  int cntneighs = 0;
    	  /* neighbors within the central bin itself */
	  int* RESTRICT ctratoms = &bins[center * atoms_per_bin];
	  
	  for (l = 0; l < bincount[center]; ++l) {
	    const int a1 = ctratoms[l];
	    int* RESTRICT neighptr = &neighbors[a1 * maxneighs ];
	  #pragma omp critical
           cntneighs = numneigh[a1];
	    const MMD_float xtmp = x[a1 * PAD + 0];
	    const MMD_float ytmp = x[a1 * PAD + 1];
	    const MMD_float ztmp = x[a1 * PAD + 2];
	    
	    for (m = l+1; m < bincount[center]; ++m) {
	      const int a2 = ctratoms[m];
	      if (a1 < nlocal || a2 < nlocal) {
		const MMD_float delx = xtmp - x[a2 * PAD + 0];
		const MMD_float dely = ytmp - x[a2 * PAD + 1];
		const MMD_float delz = ztmp - x[a2 * PAD + 2];
		const MMD_float rsq = delx * delx + dely * dely + delz * delz;
		
		if((rsq <= cutneighsq)) {
		  neighptr[cntneighs++] = a2;
//		  cn++;
	     //if (ncalls == 1 && threads->mpi_me == 0)
	     // printf("local %5d \n",a1);
	    //  printf("%5d  %5d  %5d\n", a1, a2, cntneighs);
           //   numneigh[a1] = cntneighs;
		}
	      }
	     }
	   //printf("a1: %5d, num : %5d\n", a1, cntneighs) ;
	    
	   #pragma omp critical
           numneigh[a1] = cntneighs;
	  }
	  
	  /* tower-plate neighbors */
	  for (p = 0; p < npair_stencil; p+=2) {
	    //binpair_cp[ pre-compute closest point of bins??
	    const int bin1 = center + pair_stencil[p];
	    const int bin2 = center + pair_stencil[p+1];
	    if (!locbin[bin1] && !locbin[bin2]) // if neither bin is local 
	      continue; 
	  
	    int* RESTRICT b1atoms = &bins[bin1 * atoms_per_bin];
	    int* RESTRICT b2atoms = &bins[bin2 * atoms_per_bin];
	    //printf("\t\tbin1 - bin2 pair: %d %d\n", bin1, bin2);   
	    
	    /* pick up an atom from bin1 */
	    for (l = 0; l < bincount[bin1]; ++l) {
	      const int a1 = b1atoms[l];
	      int* RESTRICT neighptr = &neighbors[a1 * maxneighs ];
	      #pragma omp critical 
	      cntneighs = numneigh[a1] ;
	      const MMD_float xtmp = x[a1 * PAD + 0];
	      const MMD_float ytmp = x[a1 * PAD + 1];
	      const MMD_float ztmp = x[a1 * PAD + 2];
	     // printf("\tatom %d\n", a1 );
	    
	      //if (distsqr_to_cp(nbrs_cp[p], x[a1]) <= cutneigh2) {
	      /* pick up another atom from the neighbor cell */
	      for (m = 0; m < bincount[bin2]; ++m) {
		const int a2 = b2atoms[m];
		
		if (a1 < nlocal || a2 < nlocal) {
		  const MMD_float delx = xtmp - x[a2 * PAD + 0];
		  const MMD_float dely = ytmp - x[a2 * PAD + 1];
		  const MMD_float delz = ztmp - x[a2 * PAD + 2];
		  const MMD_float rsq = delx*delx + dely*dely + delz*delz;
		
		  if((rsq <= cutneighsq)) {
		    neighptr[cntneighs++] = a2;
	           // cn++;
	      //if (ncalls == 1 && threads->mpi_me == 0)
	      //printf("p %5d  %5d  %5d\n", a1, a2, cntneighs);
		  }
		} 
	      } // for - bin2 atoms
	  //printf("a1: %5d, num : %5d\n", a1, cntneighs) ;
	    #pragma omp critical 
	    numneigh[a1] = cntneighs;
	    } // for - bin1 atoms
	  } // for - bin pairs based on stencil
	  
	 // numneigh[center] = cntneighs;
	}// for - local bins
  
    double welapsed = omp_get_wtime() - wstart;
    
#pragma omp barrier
    #pragma omp master
    {
      int totneigh = 0;
      printf("neighbor generation took %.4f seconds\n", welapsed);
	for( int i = 0; i< nmax; i++)
	  totneigh +=numneigh[i];
      printf("total number of neighbors is %d\n", totneigh);
      //if (threads->mpi_me == 0) fclose(f);
    }
}
  else {
  /* perform the old neighbor build */  
  /* extend atom arrays if necessary */    
  #pragma omp master
  if(nall > nmax) {
    nmax = nall;

#ifdef ALIGNMALLOC
    if(numneigh) _mm_free(numneigh);
    if(neighbors) _mm_free(neighbors);	
    numneigh = (int*) 
      _mm_malloc(nmax * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
    // hma - sizeof(int*) below, a bug?
    neighbors = (int*) 
      _mm_malloc(nmax * maxneighs * sizeof(int*) + ALIGNMALLOC, ALIGNMALLOC);
#else
    if(numneigh) free(numneigh);
    if(neighbors) free(neighbors);
    numneigh = (int*) malloc(nmax * sizeof(int));
    // hma - sizeof(int*) below, a bug?
    neighbors = (int*) malloc(nmax * maxneighs * sizeof(int*));
#endif	
  }

  /* loop over each atom, storing neighbors */
  resize = 1;
  #pragma omp barrier

  while (resize) {
    #pragma omp barrier
    int new_maxneighs = maxneighs;
    resize = 0;
    #pragma omp barrier

    OMPFORSCHEDULE
    for (int i = 0; i < nlocal; i++) {
      int* RESTRICT neighptr = &neighbors[i * maxneighs];

      /* if necessary, goto next page and add pages */
      int n = 0;
	
      const MMD_float xtmp = x[i * PAD + 0];
      const MMD_float ytmp = x[i * PAD + 1];
      const MMD_float ztmp = x[i * PAD + 2];
      
#ifdef KNC_NEIGHBUILD_INTRINSIC_SWGS 	  
      __m512d z_xtmp = _mm512_extload_pd(&x[PAD*i+0], _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8,_MM_HINT_NONE);
      __m512d z_ytmp = _mm512_extload_pd(&x[PAD*i+1], _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8,_MM_HINT_NONE);
      __m512d z_ztmp = _mm512_extload_pd(&x[PAD*i+2], _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8,_MM_HINT_NONE);	
      __m512i z_i = _mm512_extload_epi32(&i, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16,_MM_HINT_NONE);
#endif	  
      
      /* loop over atoms in i's bin */
      const int ibin = coord2bin(xtmp, ytmp, ztmp);
      
      for (int k = 0; k < nstencil; k++) {
	const int jbin = ibin + stencil[k];
	
	int* RESTRICT loc_bin = &bins[jbin * atoms_per_bin];
	
	if (ibin == jbin) {
#ifndef KNC_NEIGHBUILD_INTRINSIC_SWGS
	  for (int m = 0; m < bincount[jbin]; m++) {
            const int j = loc_bin[m];

            // for same bin as atom i skip j if i==j and 
	    // skip atoms "below and to the left" if using halfneighborlists
            if(((j == i) || (halfneigh && !ghost_newton && (j < i)) ||
                (halfneigh && ghost_newton && 
		 ((j < i) || ((j >= nlocal) && 
			      ((x[j*PAD + 2] < ztmp) || 
			       (x[j*PAD + 2] == ztmp && x[j*PAD + 1] < ytmp) ||
			       (x[j*PAD + 2] == ztmp && x[j*PAD + 1] == ytmp &&
				x[j * PAD + 0] < xtmp))))))) 
	      continue;		
	    
            const MMD_float delx = xtmp - x[j * PAD + 0];
            const MMD_float dely = ytmp - x[j * PAD + 1];
            const MMD_float delz = ztmp - x[j * PAD + 2];
            const MMD_float rsq = delx * delx + dely * dely + delz * delz;
            if ((rsq <= cutneighsq)) {
	      neighptr[n++] = j;
	      //if (ncalls == 1 && threads->mpi_me == 0)
	      //fprintf(f,"%5d  %5d  %.3f\n", MIN(i,j), MAX(i,j), rsq);
	    }
	    
          } //for
#else
	  const int bincountVal = bincount[jbin];
	  const int bincountValR = bincountVal % 8;
	  const int bincountValL = bincountVal - bincountValR; 
	  int m = 0;
	  
	  for (m=0; m < bincountValL; m+=8) {
	    const int j0 = loc_bin[m+0];
	    const int j1 = loc_bin[m+1];
	    const int j2 = loc_bin[m+2];
	    const int j3 = loc_bin[m+3];
	    const int j4 = loc_bin[m+4];
	    const int j5 = loc_bin[m+5];
	    const int j6 = loc_bin[m+6]; 
	    const int j7 = loc_bin[m+7];
	    
	    __m512i z_j = _mm512_mask_extloadunpacklo_epi32(_mm512_undefined_epi32(), kFF, &loc_bin[m+0], _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);		
	    z_j = _mm512_mask_extloadunpackhi_epi32(z_j, kFF, &loc_bin[m+16], _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);

	    __m512d j04_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);		
	    j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    
	    __m512d j15_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz,k70,  &x[PAD*j5+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    
	    __m512d j26_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz,k70, &x[PAD*j6+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	    __m512d j37_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz,k70, &x[PAD*j7+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
					
	    __m512i j04_26_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j04_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j26_xyz), _MM_PERM_CCAA);
	    __m512i j15_37_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j15_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j37_xyz), _MM_PERM_CCAA);
	    __m512i j04_26_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j26_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j04_xyz), _MM_PERM_DDBB);
	    __m512i j15_37_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j37_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j15_xyz), _MM_PERM_DDBB);
	    
	    __m512d j04152637_x =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_xyi, _mm512_int2mask(0xAA), j15_37_xyi, _MM_SWIZ_REG_CDAB));
	    __m512d j04152637_y =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j15_37_xyi, _mm512_int2mask(0x55), j04_26_xyi, _MM_SWIZ_REG_CDAB));
	    __m512d j04152637_z =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_zi, _mm512_int2mask(0xAA), j15_37_zi, _MM_SWIZ_REG_CDAB));

	    __mmask16 k_jNEQi = _mm512_mask_cmp_epi32_mask(_mm512_int2mask(0x00FF), z_j, z_i, _MM_CMPINT_NE);

	    __m512d z_delx = _mm512_sub_pd(z_xtmp, j04152637_x);
	    __m512d z_dely = _mm512_sub_pd(z_ytmp, j04152637_y);
	    __m512d z_delz = _mm512_sub_pd(z_ztmp, j04152637_z);
		
	    __m512d z_rsqx = _mm512_mul_pd(z_delx, z_delx);
	    __m512d z_rsqy = _mm512_mul_pd(z_dely, z_dely);
	    __m512d z_rsqz = _mm512_mul_pd(z_delz, z_delz);
	    
	    __m512d z_rsq = _mm512_add_pd(z_rsqx, z_rsqy);
	    z_rsq = _mm512_add_pd(z_rsq, z_rsqz); 
	    
	    __mmask8 k_rsqLTcutforcesq = _mm512_mask_cmplt_pd_mask(k_jNEQi, z_rsq,z_cutforcesq);
	    
	    unsigned int mask_k_rsqLTcutforcesq = _mm512_mask2int(k_rsqLTcutforcesq);			
	    _mm512_mask_extpackstorelo_epi32(&neighptr[n], k_rsqLTcutforcesq, z_j, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
	    _mm512_mask_extpackstorehi_epi32(&neighptr[n+16], k_rsqLTcutforcesq, z_j, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);			
	    int n_incr = _mm_popcnt_u32(mask_k_rsqLTcutforcesq);
	    n += n_incr;			
          } //for
          
	  if(bincountValR) {
	    int CMP_MASK = 0x0;
	    __m512d z_ZERO = _mm512_setzero_pd();
	    __m512d j04_xyz = z_ZERO;
	    __m512d j15_xyz = z_ZERO;
	    __m512d j26_xyz = z_ZERO;
	    __m512d j37_xyz = z_ZERO;	
	    
	    if (bincountValR == 1) {
	      CMP_MASK = 0x01;
	      const int j0 = loc_bin[m+0];	
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);					  
	    } else if (bincountValR == 2) {
	      CMP_MASK = 0x03;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];		
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);		
	    } else if (bincountValR == 3) {
	      CMP_MASK = 0x07;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];
	      const int j2 = loc_bin[m+2];
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	    } else if (bincountValR == 4) {
	      CMP_MASK = 0x0F;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];
	      const int j2 = loc_bin[m+2];
	      const int j3 = loc_bin[m+3];
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    } else if (bincountValR == 5) {
	      CMP_MASK = 0x1F;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];
	      const int j2 = loc_bin[m+2];
	      const int j3 = loc_bin[m+3];
	      const int j4 = loc_bin[m+4];
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    } else if (bincountValR == 6) {
	      CMP_MASK = 0x3F;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];
	      const int j2 = loc_bin[m+2];
	      const int j3 = loc_bin[m+3];
	      const int j4 = loc_bin[m+4];
	      const int j5 = loc_bin[m+5];
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz,k70,  &x[PAD*j5+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    } else if (bincountValR == 7) {
	      CMP_MASK = 0x7F;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];
	      const int j2 = loc_bin[m+2];
	      const int j3 = loc_bin[m+3];
	      const int j4 = loc_bin[m+4];
	      const int j5 = loc_bin[m+5];
	      const int j6 = loc_bin[m+6];
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz,k70,  &x[PAD*j5+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz,k70, &x[PAD*j6+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    } //if k remainder loop
	    
	    __m512i z_j = _mm512_mask_extloadunpacklo_epi32(_mm512_undefined_epi32(), _mm512_int2mask(CMP_MASK), &loc_bin[m+0], _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);		
	    z_j = _mm512_mask_extloadunpackhi_epi32(z_j, _mm512_int2mask(CMP_MASK), &loc_bin[m+16], _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
	    
	    __m512i j04_26_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j04_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j26_xyz), _MM_PERM_CCAA);
	    __m512i j15_37_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j15_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j37_xyz), _MM_PERM_CCAA);
	    __m512i j04_26_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j26_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j04_xyz), _MM_PERM_DDBB);
	    __m512i j15_37_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j37_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j15_xyz), _MM_PERM_DDBB);
	    
	    __m512d j04152637_x =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_xyi, _mm512_int2mask(0xAA), j15_37_xyi, _MM_SWIZ_REG_CDAB));
	    __m512d j04152637_y =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j15_37_xyi, _mm512_int2mask(0x55), j04_26_xyi, _MM_SWIZ_REG_CDAB));
	    __m512d j04152637_z =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_zi, _mm512_int2mask(0xAA), j15_37_zi, _MM_SWIZ_REG_CDAB));		  
			
	    __mmask16 k_jNEQi = _mm512_mask_cmp_epi32_mask(_mm512_int2mask(CMP_MASK), z_j, z_i, _MM_CMPINT_NE);
	    
	    __m512d z_delx = _mm512_mask_sub_pd(z_ZERO, _mm512_int2mask(CMP_MASK), z_xtmp, j04152637_x);
	    __m512d z_dely = _mm512_mask_sub_pd(z_ZERO, _mm512_int2mask(CMP_MASK), z_ytmp, j04152637_y);
	    __m512d z_delz = _mm512_mask_sub_pd(z_ZERO, _mm512_int2mask(CMP_MASK), z_ztmp, j04152637_z);
	    
	    __m512d z_rsqx = _mm512_mul_pd(z_delx, z_delx);
	    __m512d z_rsqy = _mm512_mul_pd(z_dely, z_dely);
	    __m512d z_rsqz = _mm512_mul_pd(z_delz, z_delz);
	    
	    __m512d z_rsq = _mm512_add_pd(z_rsqx, z_rsqy);
	    z_rsq = _mm512_add_pd(z_rsq, z_rsqz); 
	    
	    __mmask8 k_rsqLTcutforcesq = _mm512_mask_cmplt_pd_mask(k_jNEQi, z_rsq,z_cutforcesq);
	    unsigned int mask_k_rsqLTcutforcesq = _mm512_mask2int(k_rsqLTcutforcesq);			
	    int n_incr = _mm_popcnt_u32(mask_k_rsqLTcutforcesq);
	    
	    _mm512_mask_extpackstorelo_epi32(&neighptr[n], k_rsqLTcutforcesq, z_j, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
	    _mm512_mask_extpackstorehi_epi32(&neighptr[n+16], k_rsqLTcutforcesq, z_j, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);						
	    
	    n += n_incr;					  
	  } //if(bincountValR)
#endif
     	} 
	else {  
#ifndef KNC_NEIGHBUILD_INTRINSIC_SWGS
          for(int m = 0; m < bincount[jbin]; m++) {
            const int j = loc_bin[m];
	    
            if(halfneigh && !ghost_newton && (j < i))  continue;
	 
	    
            const MMD_float delx = xtmp - x[j * PAD + 0];
            const MMD_float dely = ytmp - x[j * PAD + 1];
            const MMD_float delz = ztmp - x[j * PAD + 2];
            const MMD_float rsq = delx * delx + dely * dely + delz * delz;
	    
            if((rsq <= cutneighsq)) {
	      neighptr[n++] = j;
	      //if (ncalls == 1 && threads->mpi_me == 0)
	      //fprintf(f,"%5d  %5d  %.3f\n", MIN(i,j), MAX(i,j), rsq);
	    }
          }
#else
	  const int bincountVal = bincount[jbin];
	  const int bincountValR = bincountVal % 8;
	  const int bincountValL = bincountVal - bincountValR; 
	  int m = 0;
	  
	  for(m=0; m < bincountValL; m+=8) {
	    
	    const int j0 = loc_bin[m+0];
	    const int j1 = loc_bin[m+1];
	    const int j2 = loc_bin[m+2];
	    const int j3 = loc_bin[m+3];
	    const int j4 = loc_bin[m+4];
	    const int j5 = loc_bin[m+5];
	    const int j6 = loc_bin[m+6]; 
	    const int j7 = loc_bin[m+7];
	    
	    __m512i z_j = _mm512_mask_extloadunpacklo_epi32(_mm512_undefined_epi32(), kFF, &loc_bin[m+0], _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);		
	    z_j = _mm512_mask_extloadunpackhi_epi32(z_j, kFF, &loc_bin[m+16], _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
	    
	    __m512d j04_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);		
	    j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
			
	    __m512d j15_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz,k70,  &x[PAD*j5+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	    __m512d j26_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz,k70, &x[PAD*j6+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    
	    __m512d j37_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz,k70, &x[PAD*j7+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    
	    __m512i j04_26_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j04_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j26_xyz), _MM_PERM_CCAA);
	    __m512i j15_37_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j15_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j37_xyz), _MM_PERM_CCAA);
	    __m512i j04_26_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j26_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j04_xyz), _MM_PERM_DDBB);
	    __m512i j15_37_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j37_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j15_xyz), _MM_PERM_DDBB);
	    
	    __m512d j04152637_x =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_xyi, _mm512_int2mask(0xAA), j15_37_xyi, _MM_SWIZ_REG_CDAB));
	    __m512d j04152637_y =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j15_37_xyi, _mm512_int2mask(0x55), j04_26_xyi, _MM_SWIZ_REG_CDAB));
	    __m512d j04152637_z =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_zi, _mm512_int2mask(0xAA), j15_37_zi, _MM_SWIZ_REG_CDAB));
	    
	    __m512d z_delx = _mm512_sub_pd(z_xtmp, j04152637_x);
	    __m512d z_dely = _mm512_sub_pd(z_ytmp, j04152637_y);
	    __m512d z_delz = _mm512_sub_pd(z_ztmp, j04152637_z);

	    __m512d z_rsqx = _mm512_mul_pd(z_delx, z_delx);
	    __m512d z_rsqy = _mm512_mul_pd(z_dely, z_dely);
	    __m512d z_rsqz = _mm512_mul_pd(z_delz, z_delz);
	    
	    __m512d z_rsq = _mm512_add_pd(z_rsqx, z_rsqy);
	    z_rsq = _mm512_add_pd(z_rsq, z_rsqz); 
	    
	    __mmask8 k_rsqLTcutforcesq = _mm512_cmplt_pd_mask(z_rsq,z_cutforcesq);      
	    unsigned int mask_k_rsqLTcutforcesq = _mm512_mask2int(k_rsqLTcutforcesq);			
	    _mm512_mask_extpackstorelo_epi32(&neighptr[n], k_rsqLTcutforcesq, z_j, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
	    _mm512_mask_extpackstorehi_epi32(&neighptr[n+16], k_rsqLTcutforcesq, z_j, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);			
	    int n_incr = _mm_popcnt_u32(mask_k_rsqLTcutforcesq);
	    n += n_incr;			
          } //for
          
	  if(bincountValR) {
	    int CMP_MASK = 0x0;
	    __m512d z_ZERO = _mm512_setzero_pd();
	    __m512d j04_xyz = z_ZERO;
	    __m512d j15_xyz = z_ZERO;
	    __m512d j26_xyz = z_ZERO;
	    __m512d j37_xyz = z_ZERO;	
	    
	    if (bincountValR == 1) {
	      CMP_MASK = 0x01;
	      const int j0 = loc_bin[m+0];	
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);					  
	    } else if (bincountValR == 2) {
	      CMP_MASK = 0x03;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];		
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);		
	    } else if (bincountValR == 3) {
	      CMP_MASK = 0x07;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];
	      const int j2 = loc_bin[m+2];
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	    } else if (bincountValR == 4) {
	      CMP_MASK = 0x0F;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];
	      const int j2 = loc_bin[m+2];
	      const int j3 = loc_bin[m+3];
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    } else if (bincountValR == 5) {
	      CMP_MASK = 0x1F;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];
	      const int j2 = loc_bin[m+2];
	      const int j3 = loc_bin[m+3];
	      const int j4 = loc_bin[m+4];
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    } else if (bincountValR == 6) {
	      CMP_MASK = 0x3F;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];
	      const int j2 = loc_bin[m+2];
	      const int j3 = loc_bin[m+3];
	      const int j4 = loc_bin[m+4];
	      const int j5 = loc_bin[m+5];
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz,k70,  &x[PAD*j5+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    } else if (bincountValR == 7) {
	      CMP_MASK = 0x7F;
	      const int j0 = loc_bin[m+0];
	      const int j1 = loc_bin[m+1];
	      const int j2 = loc_bin[m+2];
	      const int j3 = loc_bin[m+3];
	      const int j4 = loc_bin[m+4];
	      const int j5 = loc_bin[m+5];
	      const int j6 = loc_bin[m+6];
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz,k70,  &x[PAD*j5+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz,k70, &x[PAD*j6+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	      
	      j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	    } //if k remainder loop
	    
	    __m512i z_j = _mm512_mask_extloadunpacklo_epi32(_mm512_undefined_epi32(), _mm512_int2mask(CMP_MASK), &loc_bin[m+0], _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);		
	    z_j = _mm512_mask_extloadunpackhi_epi32(z_j, _mm512_int2mask(CMP_MASK), &loc_bin[m+16], _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
	    
	    __m512i j04_26_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j04_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j26_xyz), _MM_PERM_CCAA);
	    __m512i j15_37_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j15_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j37_xyz), _MM_PERM_CCAA);
	    __m512i j04_26_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j26_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j04_xyz), _MM_PERM_DDBB);
	    __m512i j15_37_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j37_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j15_xyz), _MM_PERM_DDBB);
	    
	    __m512d j04152637_x =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_xyi, _mm512_int2mask(0xAA), j15_37_xyi, _MM_SWIZ_REG_CDAB));
	    __m512d j04152637_y =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j15_37_xyi, _mm512_int2mask(0x55), j04_26_xyi, _MM_SWIZ_REG_CDAB));
	    __m512d j04152637_z =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_zi, _mm512_int2mask(0xAA), j15_37_zi, _MM_SWIZ_REG_CDAB));		  
	    
	    __m512d z_delx = _mm512_mask_sub_pd(z_ZERO, _mm512_int2mask(CMP_MASK), z_xtmp, j04152637_x);
	    __m512d z_dely = _mm512_mask_sub_pd(z_ZERO, _mm512_int2mask(CMP_MASK), z_ytmp, j04152637_y);
	    __m512d z_delz = _mm512_mask_sub_pd(z_ZERO, _mm512_int2mask(CMP_MASK), z_ztmp, j04152637_z);
			  
	    __m512d z_rsqx = _mm512_mul_pd(z_delx, z_delx);
	    __m512d z_rsqy = _mm512_mul_pd(z_dely, z_dely);
	    __m512d z_rsqz = _mm512_mul_pd(z_delz, z_delz);
					  
	    __m512d z_rsq = _mm512_add_pd(z_rsqx, z_rsqy);
	    z_rsq = _mm512_add_pd(z_rsq, z_rsqz); 
	    
	    __mmask8 k_rsqLTcutforcesq = _mm512_mask_cmplt_pd_mask(_mm512_int2mask(CMP_MASK), z_rsq,z_cutforcesq);			  
	    unsigned int mask_k_rsqLTcutforcesq = _mm512_mask2int(k_rsqLTcutforcesq);			
	    int n_incr = _mm_popcnt_u32(mask_k_rsqLTcutforcesq);
	    
	    _mm512_mask_extpackstorelo_epi32(&neighptr[n], k_rsqLTcutforcesq, z_j, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
	    _mm512_mask_extpackstorehi_epi32(&neighptr[n+16], k_rsqLTcutforcesq, z_j, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
	    
	    n += n_incr;					  
	  } //if(bincountValR)
#endif
	  
        } //if(ibin == jbin)
        
      }
      
      numneigh[i] = n;
      
      // hma: bug? if this is the last atom and n>maxneighs,
      // this code may have run out of bounds!!! 
      if (n >= maxneighs) {
        resize = 1;
        if (n >= new_maxneighs) new_maxneighs = n;
      }
    }
    
    // #pragma omp barrier
    
    if (resize) {
      #pragma omp master
      {
        maxneighs = new_maxneighs * 1.2;
#ifdef ALIGNMALLOC
	_mm_free(neighbors);
	neighbors = (int*) 
	  _mm_malloc(nmax* maxneighs * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
#else		
	free(neighbors);
        neighbors = (int*) malloc(nmax* maxneighs * sizeof(int));
#endif		
      }
#pragma omp barrier
    }
  }

  // hma: for debugging
  #pragma omp master
  {
    int def_totalneighs = 0;
    for (int i =  0; i < nlocal; ++i)
      def_totalneighs += numneigh[i];
    printf("default method totalneighs = %d\n", def_totalneighs);
    //if (ncalls == 1 && threads->mpi_me == 0)
    //  fclose(f);
  }
  
  } 

}


void Neighbor::binatoms(Atom &atom, MMD_int count)
{
  const int omp_me = omp_get_thread_num();
  const int num_omp_threads = threads->omp_num_threads;

  const int nlocal = atom.nlocal;
  const int nall = count<0?atom.nlocal + atom.nghost:count;
  const MMD_float* x = &atom.x[0][0];

  xprd = atom.box.xprd;
  yprd = atom.box.yprd;
  zprd = atom.box.zprd;

  resize = 1;

  #pragma omp barrier

  while(resize > 0) {
    #pragma omp barrier
    resize = 0;
    #pragma omp barrier

    #pragma omp for schedule(static)
    for (int i = 0; i < mbins; i++) bincount[i] = 0;


    OMPFORSCHEDULE
    for (int i = 0; i < nall; i++) {
      const int ibin = 
	coord2bin(x[i * PAD + 0], x[i * PAD + 1], x[i * PAD + 2]);

      if (bincount[ibin] < atoms_per_bin) {
        int ac;
#ifdef OpenMP31
        #pragma omp atomic capture
        ac = bincount[ibin]++;
#else
        ac = __sync_fetch_and_add(bincount + ibin, 1);
#endif
        bins[ibin * atoms_per_bin + ac] = i;
      } 
      else resize = 1;
    }
    
    // #pragma omp barrier
    
    #pragma omp master

    if(resize) {
      atoms_per_bin *= 2;
#ifdef ALIGNMALLOC
      _mm_free(bins);
      bins = (int*) _mm_malloc(mbins * atoms_per_bin * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
#else	  
      free(bins);
      bins = (int*) malloc(mbins * atoms_per_bin * sizeof(int));
#endif	  
    }
    
    // #pragma omp barrier
  }

  #pragma omp barrier
}

/* convert xyz atom coords into local bin #
   take special care to insure ghost atoms with
   coord >= prd or coord < 0.0 are put in correct bins */

inline int Neighbor::coord2bin(MMD_float x, MMD_float y, MMD_float z)
{
  int ix, iy, iz;

  if(x >= xprd)
    ix = (int)((x - xprd) * bininvx) + nbinx - mbinxlo;
  else if(x >= 0.0)
    ix = (int)(x * bininvx) - mbinxlo;
  else
    ix = (int)(x * bininvx) - mbinxlo - 1;

  if(y >= yprd)
    iy = (int)((y - yprd) * bininvy) + nbiny - mbinylo;
  else if(y >= 0.0)
    iy = (int)(y * bininvy) - mbinylo;
  else
    iy = (int)(y * bininvy) - mbinylo - 1;

  if(z >= zprd)
    iz = (int)((z - zprd) * bininvz) + nbinz - mbinzlo;
  else if(z >= 0.0)
    iz = (int)(z * bininvz) - mbinzlo;
  else
    iz = (int)(z * bininvz) - mbinzlo - 1;

  return (iz * mbiny * mbinx + iy * mbinx + ix + 1);
}



/* tower-plate */
int Neighbor::estimate_maxneighs_TP(Atom &atom)
{
  const MMD_float* x = &(atom.x[0][0]);
  const int nlocal = atom.nlocal;
  const int nall = atom.nlocal + atom.nghost;
  const int omp_me = omp_get_thread_num();

  //MMD_float xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  double wstart, welapsed, wthread;
  //int  *b1atoms, *b2atoms;
  int  i, j, k, l, m, p;
  //int  a1, a2;
  int  cntneighs;
  //int  bin1, bin2, center;
  FILE *f;

   int * tempnumneighs;
   int numatom;
  //#pragma omp master
  //if (threads->mpi_me == 0)
  //  f = fopen("myneighs.out", "w");
  
//  printf("omp_me = %d\n", omp_me);
  
  wstart = omp_get_wtime();
  
  maxnbrs[omp_me] = 0;
  totalnbrs[omp_me] = 0;
 // if ((hma == 4 ) || (hma == 5)){
//	tempnumneighs = &tempnumneigh[omp_me * nall];
        
  //    }
#pragma omp for collapse(3) schedule(guided) 
  /* grab the central bin */
  for (i = locbinxlo; i <= locbinxhi; ++i) 
    for (j = locbinylo; j <= locbinyhi; ++j)
      for (k = locbinzlo; k <= locbinzhi; ++k) {
	cntneighs = 0;
	const int center = k * mbiny * mbinx + j * mbinx + i + 1;
   if ((hma == 2)||(hma == 3)){
	#pragma omp critical
        cbin[numcbin++] = center;
   }
	/* neighbors within the central bin itself */
        int* RESTRICT ctratoms = &bins[center * atoms_per_bin];
	
	for (l = 0; l < bincount[center]; ++l) {
	  const int a1 = ctratoms[l];
	 // if ((hma == 4 )||(hma == 5)) { 
	//	numatom = 0;
	  // }
	  const MMD_float xtmp = x[a1 * PAD + 0];
	  const MMD_float ytmp = x[a1 * PAD + 1];
	  const MMD_float ztmp = x[a1 * PAD + 2];
	  
	  for (m = l+1; m < bincount[center]; ++m) {
	    const int a2 = ctratoms[m];
	    
	    if (a1 < nlocal || a2 < nlocal) {
	      const MMD_float delx = xtmp - x[a2 * PAD + 0];
	      const MMD_float dely = ytmp - x[a2 * PAD + 1];
	      const MMD_float delz = ztmp - x[a2 * PAD + 2];
	      const MMD_float rsq = delx * delx + dely * dely + delz * delz;
	      
	      if((rsq <= cutneighsq)){
		 ++cntneighs;
	    //    if ((hma == 4 )||(hma == 5)) { 
	//	  numatom++;
	  //     }
		}
	    }
	  }
	 // if ((hma == 4 )||(hma == 5)) { 
	//	tempnumneighs[a1] = numatom;
	  // }
	}
	
	/* tower-plate neighbors */
	for (p = 0; p < npair_stencil; p+=2) {
	  //binpair_cp[ pre-compute closest point of bins??
	  const int bin1 = center + pair_stencil[p];
	  const int bin2 = center + pair_stencil[p+1];
	  if (!locbin[bin1] && !locbin[bin2]) // if neither bin is local 
	    continue; 
	  
	  int* RESTRICT b1atoms = &bins[bin1 * atoms_per_bin];
	  int* RESTRICT b2atoms = &bins[bin2 * atoms_per_bin];
	 // printf("\t\tbin1 - bin2 pair: %d %d\n", bin1, bin2);   
	  	  
	  /* pick up an atom from bin1 */
	  for (l = 0; l < bincount[bin1]; ++l) {
	    const int a1 = b1atoms[l];
	  	//if ((hma == 4 )||(hma == 5)) { 
		//	numatom = 0;//tempnumneighs[a1];
		  // }
	    const MMD_float xtmp = x[a1 * PAD + 0];
	    const MMD_float ytmp = x[a1 * PAD + 1];
	    const MMD_float ztmp = x[a1 * PAD + 2];
	     //printf("\tatom %d\n", a1 );
	    
	    //if (distsqr_to_cp(nbrs_cp[p], x[a1]) <= cutneigh2) {
	    /* pick up another atom from the neighbor cell */
	    for (m = 0; m < bincount[bin2]; ++m) {
	      const int a2 = b2atoms[m];
	      
	      if (a1 < nlocal || a2 < nlocal) {
		const MMD_float delx = xtmp - x[a2 * PAD + 0];
		const MMD_float dely = ytmp - x[a2 * PAD + 1];
		const MMD_float delz = ztmp - x[a2 * PAD + 2];
		const MMD_float rsq = delx * delx + dely * dely + delz * delz;
		
		if((rsq <= cutneighsq)) {
		  ++cntneighs;
	        //	if ((hma == 4 )||(hma == 5)) { 
		//	        numatom++;
		  //      }
		  //if (threads->mpi_me == 0)
		  //fprintf(f,"%5d  %5d  %.3f\n", MIN(a1,a2), MAX(a1,a2), rsq);
		 }
	      } 
	    } // for - bin2 atoms
	  /*	if ((hma == 4 )||(hma == 5)) { 
			 tempnumneighs[a1] = numatom ; 
		   }*/
	  } // for - bin1 atoms
	} // for - bin pairs based on stencil

	if (cntneighs > maxnbrs[omp_me]) {
	  //printf("current maxneighs is %d\n", cntneighs);
	  maxnbrs[omp_me] = cntneighs;
	}
	
	totalnbrs[omp_me] += cntneighs;
      }// for - local bins

  welapsed = omp_get_wtime() - wstart;
  
  #pragma omp master
  {
    totalneighs = 0;
    for (i = 0; i < threads->omp_num_threads; ++i) {
      if (maxnbrs[i] > maxneighs) maxneighs = maxnbrs[i];
      totalneighs += totalnbrs[i];
    }
      //for (int ai = 0; ai< nall; ai++)
      //for (i = 0; i < threads->omp_num_threads; ++i)      
        // numneigh[ai] +=  tempnumneigh[i*nall + ai];
    
    printf("neighbor estimation took %.4f seconds\n", welapsed);
    printf("total estimated number of neighbors is %d\n", totalneighs);
    //if (threads->mpi_me == 0) fclose(f);
  }
  
  return (int)(maxneighs * 1.2);
}



/*
setup neighbor binning parameters
bin numbering is global: 0 = 0.0 to binsize
                         1 = binsize to 2*binsize
                         nbin-1 = prd-binsize to binsize
                         nbin = prd to prd+binsize
                         -1 = -binsize to 0.0
coord = lowest and highest values of ghost atom coords I will have
        add in "small" for round-off safety
mbinlo = lowest global bin any of my ghost atoms could fall into
mbinhi = highest global bin any of my ghost atoms could fall into
mbin = number of bins I need in a dimension
stencil() = bin offsets in 1-d sense for stencil of surrounding bins
*/

int Neighbor::setup(Atom &atom)
{
  int i, j, k, stmax;
  MMD_float coord;
  int mbinxhi, mbinyhi, mbinzhi;
  int nextx, nexty, nextz;
  int num_omp_threads = threads->omp_num_threads;

  if (threads->mpi_me == 0)
    fprintf(stderr, "# Neighbor Setup: \n");
  
  cutneighsq = cutneigh * cutneigh;

  xprd = atom.box.xprd;
  yprd = atom.box.yprd;
  zprd = atom.box.zprd;

  /*
  c bins must evenly divide into box size,
  c   becoming larger than cutneigh if necessary
  c binsize = 1/2 of cutoff is near optimal

  if (flag == 0) {
    nbinx = 2.0 * xprd / cutneigh;
    nbiny = 2.0 * yprd / cutneigh;
    nbinz = 2.0 * zprd / cutneigh;
    if (nbinx == 0) nbinx = 1;
    if (nbiny == 0) nbiny = 1;
    if (nbinz == 0) nbinz = 1;
  }
  */

  binsizex = xprd / nbinx;
  binsizey = yprd / nbiny;
  binsizez = zprd / nbinz;
  bininvx = 1.0 / binsizex;
  bininvy = 1.0 / binsizey;
  bininvz = 1.0 / binsizez;

  coord = atom.box.xlo - cutneigh - SMALL * xprd;
  mbinxlo = static_cast<int>(coord * bininvx);

  if(coord < 0.0) mbinxlo = mbinxlo - 1;

  coord = atom.box.xhi + cutneigh + SMALL * xprd;
  mbinxhi = static_cast<int>(coord * bininvx);

  coord = atom.box.ylo - cutneigh - SMALL * yprd;
  mbinylo = static_cast<int>(coord * bininvy);

  if(coord < 0.0) mbinylo = mbinylo - 1;

  coord = atom.box.yhi + cutneigh + SMALL * yprd;
  mbinyhi = static_cast<int>(coord * bininvy);

  coord = atom.box.zlo - cutneigh - SMALL * zprd;
  mbinzlo = static_cast<int>(coord * bininvz);

  if(coord < 0.0) mbinzlo = mbinzlo - 1;

  coord = atom.box.zhi + cutneigh + SMALL * zprd;
  mbinzhi = static_cast<int>(coord * bininvz);

  /* extend bins by 1 in each direction to insure stencil coverage */

  mbinxlo = mbinxlo - 1;
  mbinxhi = mbinxhi + 1;
  mbinx = mbinxhi - mbinxlo + 1;

  mbinylo = mbinylo - 1;
  mbinyhi = mbinyhi + 1;
  mbiny = mbinyhi - mbinylo + 1;

  mbinzlo = mbinzlo - 1;
  mbinzhi = mbinzhi + 1;
  mbinz = mbinzhi - mbinzlo + 1;

  mbins = mbinx * mbiny * mbinz; 
  
  //atoms_per_thread = (mbins/num_omp_threads) * atoms_per_bin;  // -mathi
  atoms_per_thread = mbins * atoms_per_bin;  // -mathi
//  neighs_per_thread = (mbins/num_omp_threads) * maxneighs;  // -mathi

   
  /*
  compute bin stencil of all bins whose closest corner to central bin
  is within neighbor cutoff
  for partial Newton (newton = 0),
  stencil is all surrounding bins including self
  for full Newton (newton = 1),
  stencil is bins to the "upper right" of central bin, does NOT include self
  next(xyz) = how far the stencil could possibly extend
  factor < 1.0 for special case of LJ benchmark so code will create
  correct-size stencil when there are 3 bins for every 5 lattice spacings
  */

  nextx = static_cast<int>(cutneigh * bininvx);

  if (nextx * binsizex < FACTOR * cutneigh) nextx++;

  nexty = static_cast<int>(cutneigh * bininvy);

  if (nexty * binsizey < FACTOR * cutneigh) nexty++;

  nextz = static_cast<int>(cutneigh * bininvz);

  if (nextz * binsizez < FACTOR * cutneigh) nextz++;

  stmax = (2 * nextz + 1) * (2 * nexty + 1) * (2 * nextx + 1);

  
  if (hma == 0) {
#ifdef ALIGNMALLOC
    if (stencil) _mm_free(stencil);
    stencil = (int*) _mm_malloc(stmax*sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
#else  
    if (stencil) free(stencil);
    stencil = (int*) malloc(stmax * sizeof(int));
#endif  
    
    nstencil = 0;
    int kstart = -nextz;
    
    if (halfneigh && ghost_newton) {
      kstart = 0;
      stencil[nstencil++] = 0;
    }
    
    for (k = kstart; k <= nextz; k++) {
      for (j = -nexty; j <= nexty; j++) {
	for (i = -nextx; i <= nextx; i++) {
	  if (!ghost_newton || !halfneigh || (k > 0 || j > 0 || (j==0 && i>0)))
	    if (bindist(i, j, k) < cutneighsq) {
	      stencil[nstencil++] = k * mbiny * mbinx + j * mbinx + i;
	    }
	}
      }
    }

    if (threads->mpi_me == 0)
      fprintf(stderr, "\t# done with the old setup.\n");

  }  
  else if ((hma == 1)||(hma ==4)){ /* hma */

#ifdef ALIGNMALLOC
    if (pair_stencil) _mm_free(pair_stencil);
    pair_stencil = (int*) 
      _mm_malloc(stmax * sizeof(int) * 2 + ALIGNMALLOC, ALIGNMALLOC);

    if (locbin) _mm_free(locbin);
    locbin = (char*) _mm_malloc(mbins*sizeof(char) + ALIGNMALLOC, ALIGNMALLOC);

#else  
    if (pair_stencil) free(pair_stencil);
    pair_stencil = (int*) malloc(stmax * sizeof(int) * 2);

    if (locbin) free(locbin);
    locbin = (char*) malloc(mbins * sizeof(char));
#endif
    
    /* print setup */
    if (threads->mpi_me == 0) {
      printf("# Neighbor Setup: \n");
      printf("\t# box size:  %.2lf  %.2lf  %.2lf\n", xprd, yprd, zprd);
      printf("\t# bin size:  %.2lf  %.2lf  %.2lf\n", 
	     binsizex, binsizey, binsizez);
      printf("\t# mbins  :  %d  %d  %d\n", mbinx, mbiny, mbinz);
      printf("\t# bin ext:  %d  %d  %d\n", nextx, nexty, nextz);
      printf("\t# stmax=  %d\n", stmax);
      printf("\t# mbinxlo  mbinylo  mbinzlo=  %d %d %d\n", 
	     mbinxlo, mbinylo, mbinzlo);
      printf("\t# mbinxhi  mbinyhi  mbinzhi=  %d %d %d\n", 
	     mbinxhi, mbinyhi, mbinzhi);
    }
    
    /* error check */
    if (nextx != nexty || nextx != nextz || nexty != nextz) {
      fprintf(stderr, "ERROR: Tower-Plate (TP) stencil requires" 
	      "nextx = nexty = nextz. Current binsizes and extents are:"
	      "\tbin sizes  = %.2lf  %.2lf  %.2lf\n"
	      "\tbin extents= %d  %d  %d\n",
	      binsizex, binsizey, binsizez, nextx, nexty, nextz);
      exit(1);
    }
    
    /* when nextx is 1, there are at most 2x5 = 10 colors,
       when nextx is 2, there are at most 3x10 = 30 colors
       when nextx is 3, there are at most 4x15 = 60 colors */
    max_colors = 5 * nextx * (nextx + 1);
    pat = 5 * nextx;
    
    locbinxlo = (int)(atom.box.xlo * bininvx) - mbinxlo;
    locbinxhi = (int)(atom.box.xhi * bininvx) - mbinxlo;
    int nlocbinx = locbinxhi - locbinxlo + 1;
    
    locbinylo = (int)(atom.box.ylo * bininvy) - mbinylo;
    locbinyhi = (int)(atom.box.yhi * bininvy) - mbinylo;
    int nlocbiny = locbinyhi - locbinylo + 1;
    
    locbinzlo = (int)(atom.box.zlo * bininvz) - mbinzlo;
    locbinzhi = (int)(atom.box.zhi * bininvz) - mbinzlo;
    int nlocbinz = locbinzhi - locbinzlo + 1;
    
    int max_same = 
      (nlocbinx/(nextx+1) + 1) * (nlocbiny/pat + 1) * (nlocbinz/pat + 1) * pat;

    if (threads->mpi_me == 0) {    
      printf("\t# xlo xhi nlocbinx:  %d %d %d\n", 
	     locbinxlo, locbinxhi, nlocbinx);
      printf("\t# ylo zhi nlocbiny:  %d %d %d\n", 
	     locbinylo, locbinyhi, nlocbiny);
      printf("\t# zlo zhi nlocbinz:  %d %d %d\n", 
	     locbinzlo, locbinzhi, nlocbinz);
      printf("\t# max_same: %d\n", max_same);
    }    
    
    
    /* allocate space for the work packets */
    num_cbins = (int*) calloc(max_colors, sizeof(int));
    cbin_list = (int**) calloc(max_colors, sizeof(int*));
    //num_cintrs = (int***) calloc(max_colors, sizeof(int**));
    //colors = (lj_data***) calloc(cmax, sizeof(lj_data**));
    
    for (i = 0; i < max_colors; ++i) {
      cbin_list[i] = (int*) calloc(max_same, sizeof(int));
      //num_cintrs[i] = (int**) calloc(max_same, sizeof(int*));
      //colors[i] = (lj_data**) calloc(max_same, sizeof(lj_data*));
    }
    
    
    /* create the pair stencil */
    int di, dj, dk, l;
    npair_stencil = 0;
    
    /* choose a cell in the tower - dj = up/down */
    for (dj = -nexty; dj <= nexty; dj++) {
      if (dj == 0) {
	/* the cell at 0,0,0 interacts with half others in tower */
	for (l = 1; l <= nexty; l++) {
	  pair_stencil[npair_stencil] = 0; // central cell
	  pair_stencil[npair_stencil+1] = l * mbinx; // di=dk=0
	  //Find_Closest_Point( g, i, j, k, i, t, k, cp[cnt] );
	  npair_stencil+=2;
	}
      }	
      
      /* choose a cell in the half plate - di = left/right */
      /*for (di = 0; di <= nextx; di++) {
	if (di == 0) dk = 1;
	else dk = -nextz;
	
	for ( ; dk <= nextz; dk++) {
	  pair_stencil[npair_stencil] = dj * mbinx; //di=dk=0 in the tower
	  pair_stencil[npair_stencil+1] = dk * mbiny * mbinx + di;
	  //Find_Closest_Point( g, i, y, k, x, j, z, cp[cnt] );	  
          npair_stencil += 2;
	}
	}*/

      for (dk = -nextz; dk <= nextz; dk++) {
	if (dk <= 0) di = 1;
	else di = 0;
	
	for ( ; di <= nextx; di++) {
	  pair_stencil[npair_stencil] = dj * mbinx; //di=dk=0 in the tower
	  pair_stencil[npair_stencil+1] = dk * mbiny * mbinx + di;
	  //Find_Closest_Point( g, i, y, k, x, j, z, cp[cnt] );	  
          npair_stencil += 2;
	}
	}
    }
    
   /* #pragma omp master
    if (threads->mpi_me == 0) {
      printf("# Checking the pair stencil:\n"// (assume central cell 5,5,5):\n"
	     "\t#Pair      d1 -    d2\n");//  bin1-bin2\n"); 
      for (i = 0; i < npair_stencil; i+=2) 
	printf("\t  %d   %5d - %5d\n", //   (%d,%d,%d)-(%d,%d,%d)\n",
	       i/2, pair_stencil[i], pair_stencil[i+1]);
    }
    */
    
    /* color the bins */
    int map[5][20][20];
    int x, y, z, c, ci, cj, ck, top, binno;
    int yinc = nextx + 1;
    pat = 5 * nextx;
    
    /* create the pattern map */
    y = 0;
    for (z = 0; z < pat; ++z) {
      for (c = 0; c < pat; ++c) {
	for (x = 0; x < nextx+1; ++x)
	  map[x][(y+c)%pat][z] = c + x*pat;
      }
      y = (y+yinc) % pat;
    }

    //for (x = 0; nextx+1; ++x) {
    //}
    
    for (c = 0; c < max_colors; ++c) 
      num_cbins[c] = 0;

    for (i = 0; i < mbins; ++i) 
      locbin[i] = 0;
   
    if (threads->mpi_me == 0)
      printf("# Coloring the bins and marking local ones:\n"); 
    for (i = locbinxlo; i <= locbinxhi; ++i) {
      ci = i % (nextx+1);
      for (j = locbinylo; j <= locbinyhi; ++j) {
	cj = j % pat;
	for (k = locbinzlo; k <= locbinzhi; ++k) {
	  binno = k * mbiny * mbinx + j * mbinx + i + 1;
	  locbin[binno] = 1;
	  
	  ck = k % pat;	  
	  c = map[ci][cj][ck];
	  top = num_cbins[c];
	  cbin_list[c][top] = binno;
	  num_cbins[c]++;
	  
	  //if (threads->mpi_me == 0)
	  //printf("\t# bin(%d,%d,%d): binno=%d color=%d\n", i, j, k, binno,c);
	}
      }
    } 

  }
else{

  const int nall = atom.nlocal + atom.nghost;
 #ifdef ALIGNMALLOC
    if (cbin) _mm_free(cbin);  //-mathi		
    cbin = (int*)  _mm_malloc(mbins * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
    if (pair_stencil) _mm_free(pair_stencil);
    pair_stencil = (int*) 
      _mm_malloc(stmax * sizeof(int) * 2 + ALIGNMALLOC, ALIGNMALLOC);

    if (locbin) _mm_free(locbin);
    locbin = (char*) _mm_malloc(mbins*sizeof(char) + ALIGNMALLOC, ALIGNMALLOC);
    /* mathi */
    if (tempF) _mm_free(tempF);
    //printf("*****apt: %5d, not: %5d, PAD: %5d\n", atoms_per_thread, num_omp_threads ,PAD);
    tempF = (MMD_float*) _mm_malloc(atoms_per_thread * num_omp_threads * sizeof(MMD_float)*PAD+ ALIGNMALLOC, ALIGNMALLOC);
    if (tempnumneigh) _mm_free(tempnumneigh);
    tempnumneigh = (int*) _mm_malloc(nall * num_omp_threads * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
#else  
    if (pair_stencil) free(pair_stencil); 
    pair_stencil = (int*) malloc(stmax * sizeof(int) * 2);
    if (cbin) free(cbin);   //-mathi	
    cbin = (int*) malloc(mbins * sizeof(int));   

    if (locbin) free(locbin);
    locbin = (char*) malloc(mbins * sizeof(char));
    /* mathi */
    if (tempF) free(tempF);
   // printf("******apt: %5d, not: %5d, PAD: %5d\n", atoms_per_thread, num_omp_threads ,PAD);
    tempF = (MMD_float*) malloc(atoms_per_thread * num_omp_threads * sizeof(MMD_float)*PAD);
    if (tempnumneigh) free(tempnumneigh);
    tempnumneigh = (int*) malloc(nall * num_omp_threads * sizeof(int));
#endif
    
    /* print setup */
    if (threads->mpi_me == 0) {
      printf("# Neighbor Setup: \n");
      printf("\t# box size:  %.2lf  %.2lf  %.2lf\n", xprd, yprd, zprd);
      printf("\t# bin size:  %.2lf  %.2lf  %.2lf\n", 
	     binsizex, binsizey, binsizez);
      printf("\t# mbins  :  %d  %d  %d\n", mbinx, mbiny, mbinz);
      printf("\t# bin ext:  %d  %d  %d\n", nextx, nexty, nextz);
      printf("\t# stmax=  %d\n", stmax);
      printf("\t# mbinxlo  mbinylo  mbinzlo=  %d %d %d\n", 
	     mbinxlo, mbinylo, mbinzlo);
      printf("\t# mbinxhi  mbinyhi  mbinzhi=  %d %d %d\n", 
	     mbinxhi, mbinyhi, mbinzhi);
    }
    
    /* error check */
    if (nextx != nexty || nextx != nextz || nexty != nextz) {
      fprintf(stderr, "ERROR: Tower-Plate (TP) stencil requires" 
	      "nextx = nexty = nextz. Current binsizes and extents are:"
	      "\tbin sizes  = %.2lf  %.2lf  %.2lf\n"
	      "\tbin extents= %d  %d  %d\n",
	      binsizex, binsizey, binsizez, nextx, nexty, nextz);
      exit(1);
    }
    
    /* when nextx is 1, there are at most 2x5 = 10 colors,
       when nextx is 2, there are at most 3x10 = 30 colors
       when nextx is 3, there are at most 4x15 = 60 colors */
    //max_colors = 5 * nextx * (nextx + 1);
    pat = 5 * nextx;
    
    locbinxlo = (int)(atom.box.xlo * bininvx) - mbinxlo;
    locbinxhi = (int)(atom.box.xhi * bininvx) - mbinxlo;
    int nlocbinx = locbinxhi - locbinxlo + 1;
    
    locbinylo = (int)(atom.box.ylo * bininvy) - mbinylo;
    locbinyhi = (int)(atom.box.yhi * bininvy) - mbinylo;
    int nlocbiny = locbinyhi - locbinylo + 1;
    
    locbinzlo = (int)(atom.box.zlo * bininvz) - mbinzlo;
    locbinzhi = (int)(atom.box.zhi * bininvz) - mbinzlo;
    int nlocbinz = locbinzhi - locbinzlo + 1;
    

    if (threads->mpi_me == 0) {    
      printf("\t# xlo xhi nlocbinx:  %d %d %d\n", 
	     locbinxlo, locbinxhi, nlocbinx);
      printf("\t# ylo zhi nlocbiny:  %d %d %d\n", 
	     locbinylo, locbinyhi, nlocbiny);
      printf("\t# zlo zhi nlocbinz:  %d %d %d\n", 
	     locbinzlo, locbinzhi, nlocbinz);
    }    
    
    
    
    /* create the pair stencil */
    int di, dj, dk, l;
    npair_stencil = 0;
    
    /* choose a cell in the tower - dj = up/down */
    for (dj = -nexty; dj <= nexty; dj++) {
      if (dj == 0) {
	/* the cell at 0,0,0 interacts with half others in tower */
	for (l = 1; l <= nexty; l++) {
	  pair_stencil[npair_stencil] = 0; // central cell
	  pair_stencil[npair_stencil+1] = l * mbinx; // di=dk=0
	  //Find_Closest_Point( g, i, j, k, i, t, k, cp[cnt] );
	  npair_stencil+=2;
	}
      }	
      

      for (dk = -nextz; dk <= nextz; dk++) {
	if (dk <= 0) di = 1;
	else di = 0;
	
	for ( ; di <= nextx; di++) {
	  pair_stencil[npair_stencil] = dj * mbinx; //di=dk=0 in the tower
	  pair_stencil[npair_stencil+1] = dk * mbiny * mbinx + di;
	  //Find_Closest_Point( g, i, y, k, x, j, z, cp[cnt] );	  
          npair_stencil += 2;
	}
	}
    }
    
   /* #pragma omp master
    if (threads->mpi_me == 0) {
      printf("# Checking the pair stencil:\n"// (assume central cell 5,5,5):\n"
	     "\t#Pair      d1 -    d2\n");//  bin1-bin2\n"); 
      for (i = 0; i < npair_stencil; i+=2) 
	printf("\t  %d   %5d - %5d\n", //   (%d,%d,%d)-(%d,%d,%d)\n",
	       i/2, pair_stencil[i], pair_stencil[i+1]);
    }*/
    
    
    int  ci, cj, binno;
   pat = 5 * nextx;
   
    if (threads->mpi_me == 0)
     // printf("# Marking local bines:\n"); 
    for (i = locbinxlo; i <= locbinxhi; ++i) {
      ci = i % (nextx+1);
      for (j = locbinylo; j <= locbinyhi; ++j) {
	cj = j % pat;
	for (k = locbinzlo; k <= locbinzhi; ++k) {
	  binno = k * mbiny * mbinx + j * mbinx + i + 1;
	  locbin[binno] = 1;
	  
	}
      }
    }
}

  /* hma: why is it necessary to allocate bincount and bins 
     for each OMP thread?*/
#ifdef ALIGNMALLOC
  if (bincount) _mm_free(bincount);

  bincount = (int*) _mm_malloc(mbins * num_omp_threads * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);

  if (bins) _mm_free(bins);

  bins = (int*) _mm_malloc(mbins * num_omp_threads * atoms_per_bin * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
#else
  if (bincount) free(bincount);

  bincount = (int*) malloc(mbins * num_omp_threads * sizeof(int));

  if (bins) free(bins);

  bins = (int*) malloc(mbins * num_omp_threads * atoms_per_bin * sizeof(int));
#endif    
  return 0;
}

/* compute closest distance between central bin (0,0,0) and bin (i,j,k) */

MMD_float Neighbor::bindist(int i, int j, int k)
{
  MMD_float delx, dely, delz;

  if(i > 0)
    delx = (i - 1) * binsizex;
  else if(i == 0)
    delx = 0.0;
  else
    delx = (i + 1) * binsizex;

  if(j > 0)
    dely = (j - 1) * binsizey;
  else if(j == 0)
    dely = 0.0;
  else
    dely = (j + 1) * binsizey;

  if(k > 0)
    delz = (k - 1) * binsizez;
  else if(k == 0)
    delz = 0.0;
  else
    delz = (k + 1) * binsizez;

  return (delx * delx + dely * dely + delz * delz);
}

