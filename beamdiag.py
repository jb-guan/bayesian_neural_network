#!/usr/bin/env python2

"""
    The script for particle bunch diagnostics.
    Originally Written by Dr. John Farmer of MPP&CERN. Redeveloped by Linbo Liang of the University of Manchester.
"""

import numpy as np
import os, sys
import scipy.constants
from scipy.signal import convolve
#import matplotlib.pyplot as plt
import re, parse
import argparse
#from mpl_toolkits.axes_grid1 import make_axes_locatable

# user defined functions
def mean(a,weights):
  return np.average(a,weights=weights)

def meansq(a,weights):
  return np.average(a**2,weights=weights)

def var(a,weights):
  return np.average(a**2,weights=weights)-np.average(a,weights=weights)**2

def rms(a,weights):
  return np.sqrt(meansq(a,weights=weights))

def std(a,weights):
  return np.sqrt(var(a,weights=weights))

def rangeMask(a,weights=1,range=0.01,binsinrange=10):
  amax=a.max()
  amin=a.min()
  b=[amin]
  while b[-1]<amax:
    b.append(b[-1]*(1+range/binsinrange))

  ax,_=np.histogram(a,weights=weights,bins=b)

  axc=convolve(ax,np.ones(binsinrange),"same")

  bmc=np.argmax(axc)

  return (a>b[bmc-binsinrange//2])*(a<b[bmc+(binsinrange+1)//2])


# def find(cfg, par): # READING configuration file
#     ans=re.search('\s' + par + '\s?=\s?[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', cfg)
#     return float(ans.group(0).replace(par, '').replace('=', ''))
def beamdiago(filepath,phase,species):
  #filepath = 'G:\\LCODE\\makefile\\'
  # default values
  c=scipy.constants.c
  e=scipy.constants.e
  m=scipy.constants.electron_mass
  mp=scipy.constants.proton_mass
  e0=scipy.constants.epsilon_0
  pi=np.pi
  IA=4*pi*e0*m*c**3/e

  speciesname={'-1':'electron','1':'proton'}

  cropCore = 1
  graph = 0

  if phase<0:
    filename=filepath+"beamfile.bin"
  else:
    filename=filepath+"tb%s.swp" % str(phase).zfill(5)

  if species not in [-1,1]:
      sys.stderr.write("Choose correct species [1: proton; -1: witness]\nExit with error!\n")
      exit(-1)

  # read geometric parameters from the configuration file
  with open(filepath+'parameters.txt', 'r') as f:
      text = f.read()
      text = re.sub(r'#.*', "", text)  # remove comments
      text = re.sub(r'[ \t]', "", text)  # remove spaces and tabs
      Lr=parse.search("rmax={:g}", text)[0]
      Lz=parse.search("zmax={:g}", text)[0]
      dz=parse.search("dz={:g}", text)[0]
      dr=parse.search("dr={:g}", text)[0]
      dt=parse.search("dt={:g}", text)[0]
      Qw=-parse.search("Qw={:g}", text)[0]#初始电荷量
      Nw=parse.search("Nw={:g}", text)[0]#初始宏粒子数
      Nz=int(Lz/dz)
      Nr=int(Lr/dr)
      if species == -1:
          Q0=-parse.search("Qw={:g}", text)[0]/1e-12
          em0=parse.search("emw={:g}", text)[0]
          freeze=parse.search("freezew={:g}", text)[0]
      else:
          Q0=parse.search("Qp={:g}", text)[0]/1e-12
          em0=parse.search("emd={:g}", text)[0]
          freeze=parse.search("freezep={:g}", text)[0]
      n0=parse.search("n_plasma={:g}", text)[0]

  if freeze == 1:
      dummyMass=1e12
  else:
      dummyMass=1
  # if 'LCODE_ne' in os.environ:
  #   n0=float(os.environ['LCODE_ne'])
  # else:
  #   n0=7e20 #/m^3
  # if os.path.isfile(cfg_file):
  #   # print('# use configuration dz\n')
  #   dz=find(config, 'xi-step')
  # elif 'LCODE_dz' in os.environ:
  # #  print('use keyboard input dz\n')
  #   dz=float(os.environ['LCODE_dz'])
  # else:
  #   dz=0.01
  if 'LCODE_zwindow' in os.environ:
    zwindow=float(os.environ['LCODE_zwindow'])
  else:
    zwindow=0
  if 'LCODE_cropEnergy' in os.environ:
    cropEnergy=float(os.environ['LCODE_cropEnergy'])
  else:
    cropEnergy=0
  #if 'LCODE_cropCore' in os.environ:
  #  cropCore=float(os.environ['LCODE_cropCore'])
  #else:
  #  cropCore=0

  op=(n0*e**2/e0/m)**.5 #omegap
  k=op/c
  um=1e6/k


  rfac=(2*c**2*em0**2/op**2)**.5*k**2  #normalised r
  prfac=(em0**2*op**2/2/c**2)**.5


  z,r,pz,pr,pa,q,w,N=np.fromfile(filename).reshape(-1,8)[:-1].transpose()

 

  pz/=dummyMass

  sq=dummyMass**-1*species
  mw=w*(q==sq)/species # masked weight
  #print(len(q),sq,len(w),len(mw))

   ############################################
  Nwf = 0
  for i in range(len(q)):
    if q[i] == sq:
      Nwf +=1
  if species == -1:
    Q0 = Qw/Nw*Nwf*1e12 #pC
  ############################################

  if cropEnergy:
      mw=mw*rangeMask(pz,weights=mw)
  if cropCore:
      sqrtpz=pz**.5
      if species==-1:
          incore=(r**2*sqrtpz/rfac+(pr**2+(pa/r)**2)/prfac/sqrtpz<4)
      else:
          incore=1
      mwincore=mw*incore # masked weight
  
  if sum(mw) == 0 or sum(mwincore) == 0:
    return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


  mz  =mean(z,weights=mw)
  msz =meansq(z,weights=mw)
  msr =meansq(r,weights=mw)
  msrin =meansq(r,weights=mwincore)
  mpz =mean(pz,weights=mw)
  mspz=meansq(pz,weights=mw)
  mpr =mean(pr,weights=mw)
  mspr=meansq(pr,weights=mw)
  msprin=meansq(pr,weights=mwincore)
  mpa =mean(pa/r,weights=mw)
  mspa=meansq(pa/r,weights=mw)
  mspain=meansq(pa/r,weights=mwincore)

  std_z=(msz-mz**2)**.5
  rms_r=msr**.5
  std_pz=(float(mspz)-float(mpz)**2)**.5 #### new float
  std_pr=(mspr-mpr**2)**.5
  std_pa=(mspa-mpa**2)**.5

  varz = var(z, weights=mw)
  varpz= var(pz, weights=mw)

  emr  =(msr*(mspr+mspa)-mean(r*pr,weights=mw)**2)**.5/2 # from LCODE manual
  emin =(msrin*(msprin+mspain)-mean(r*pr,weights=mwincore)**2)**.5/2
  emz =(varz*varpz-mean((z-mz)*(pz-mpz), weights=mw)**2)**.5

  sumw=np.sum(mw)
  sumwin=np.sum(mwincore)

  if graph:
      fig, ax=plt.subplots(1, 1)
      plt.hist2d(z,r,bins=[Nz,Nr],weights=mw)
      plt.title(r'%s, t = %d $\omega_p^{-1}$'%(speciesname[str(species)],phase))
      plt.xlabel(r'$k_p\xi$')
      plt.ylabel(r'$k_pr$')
      # plt.xlim(None,-5)
      # plt.ylim(0,2)
      cbar=plt.colorbar()
      cbar.set_label('$n_b (a. u.)$')
      fig.tight_layout()
      fig.savefig('./nb_%s_%d.png'%(speciesname[str(species)],phase),dpi=300)
  
  #return z,r,[Nz,Nr],mw
  return [phase,  #1
  (mz+zwindow)*um,std_z*um, #2,3
  rms_r*um/2**.5,   #4
  mpz,std_pz,   #5,6
  mpr,std_pr,   #7,8
  mpa,std_pa,   #9,10
  emr*um,emz*um,    #11,12
  sumw*IA*dz/2/op*1e12,   #13
  np.count_nonzero(mw),   #14
  emin*um,    #15
  sumwin*IA*dz/2/op*1e12/Q0,    #16
  Q0]   #17

if __name__=='__main__':
  # get user input
  # cfg_file=args.cfg
  path = 'G:\\LCODE\\phase space\\'
  phase = 50000
  species = -1
  a = beamdiago(path,phase,species)

 




#1   - phase
#2,3 - <z>, z_rms (um)
#4   - r_rms/sqrt(2) (um)
#5,6 - <pz>, pz_rms
#7,8 - <pr>, pr_rms
#9,10 - <pa>, pa_rms
#11,12  - emittance (um)
#13  - weight (pC - but needs correct dz)
#13  - Nmacro
#14  - comments (optional)
