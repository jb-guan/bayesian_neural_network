#!/usr/bin/env python

"""
   A python script for generating both the driver and witness bunch distribution in the toy model for AWAKE Run 2 acceleration stage simulation.
   Please refer to Dr. Veronica Olsen's paper for the detail of the model. DOI: 10.1103/PhysRevAccelBeams.21.011301
   This script is originally written by Dr. John Farmer of MPP&CERN and redeveloped by Linbo Liang of University of Manchester with extented features.
"""
from encodings import utf_8 #new
import numpy as np
import os, sys
import re
import scipy.constants
from scipy import special
import scipy.stats as stats
from os.path import basename
import argparse
# from icecream import ic
sim_name = basename(os.getcwd())
sim_path = os.getcwd()

c=scipy.constants.c
e=scipy.constants.e
me=scipy.constants.electron_mass
mp=scipy.constants.proton_mass
e0=scipy.constants.epsilon_0
mec2 = 0.511 #[MeV]
pi=np.pi
IA=4*pi*e0*me*c**3/e  # the Alfven current IA = 17 kA

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('cfg', type=str, nargs='+', help='Path and name of the configuration file, e.g., ./input.cfg')
parser.add_argument('qe', type=float, nargs=1, help='Charge of the witness bunch. Unit in pC.')
parser.add_argument('em', type=float, nargs=1, help='Emittance of the witness bunch. Unit in um.')
parser.add_argument('sigz', type=float, nargs=1, help="RMS length of the witness bunch. Unit in um.")
parser.add_argument('xi0', type=float, nargs=1, help='Longitudinal position of the bunch in normalised unit.') #following distance of driven (normalised unit)
parser.add_argument('n0', type=float, nargs=1, help='Plasma density,unit: cm^-3/1e14')
parser.add_argument('spd', type=float, nargs=1, help='spread of driver,unit: %')
parser.add_argument('Rd', type=float, nargs=1, help='length of driver,unit: m')
parser.add_argument('Qd', type=float, nargs=1, help='charge of driver,unit: ')
parser.add_argument('Ld', type=float, nargs=1, help='radius of driver,unit: m')

args = parser.parse_args()
cfg=args.cfg[0]
qe = args.qe[0] * -1.e-12
em = args.em[0] * 1.e-6
sigz = args.sigz[0] * 1.e-6
xi0 = args.xi0[0]
n0 = args.n0[0]
spd = args.spd[0]
Ld = args.Ld[0] * 1e-6
Qd = args.Qd[0] * 1e-9
Rd = args.Rd[0] * 1e-6

print("#", args)

# get the shape factor from keyboard input
if len(sys.argv) < 1:
    sys.stderr.write("Not enough inputs! Enter 'python beamfile.py -h' for instruction.\n")
    exit(-1)

# READING configuration file
def find(cfg, par):
    ans=re.search('\s' + par + '\s?=\s?[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', cfg.decode('utf-8')) #.decode('utf-8') new
    return float(ans.group(0).replace(par, '').replace('=', ''))

# path='./'
# cfg_name='run.cfg'
config=open(cfg, 'rb').read() #r-->rb new
r_size=find(config, 'window-width')
xi_size=find(config, 'window-length')
dr=find(config, 'r-step')
dz=find(config, 'xi-step')
t_step=find(config, 'time-step')
t_max=find(config, 'time-limit')
save_beam=find(config, 'saving-period')
save_output=find(config, 'output-time-period')

# plasma parameters
n0 = n0 * 1e6 * 1e14 # plasma density [m^-3]
omegap = np.sqrt(n0 * e ** 2 / me / e0)
kp = omegap / c
lambdap = 2 * np.pi / kp


# proton bunch
#Ld=40e-6
#Rd=200e-6
z0d=-Ld*kp*5
gammad=400e9*e/mp/c**2
#Qd=2.34e-9
emd=0
dpd=0
freezd=1 # 1 for true; 0 for false

grid=1

if grid:
  ppcz=1
  ppcr=2  # should be an integer at least 2

  zds=z0d+0.5*dz/ppcz+np.arange(-int(3*Ld*kp/dz)*dz,int(3*Ld*kp/dz)*dz,dz/ppcz)
  print(e0,z0d,int(3*Ld*kp/dz)*dz,dz/ppcz)
  rds=    0.5*dr/ppcr+np.arange(0,int(5*Rd*kp/dr)*dr,dr/ppcr)
  zrd=np.array([[zi,ri] for zi in zds for ri in rds])[::-1]
  Nd=len(zrd)
  wd=2*Qd*c/IA * kp/dz * stats.norm.pdf(zrd[:,0],scale=Ld*kp,loc=z0d)*dz/ppcz * stats.weibull_min.pdf(zrd[:,1],2,scale=2**.5*Rd*kp)*dr/ppcr
#  wbd=2*Qd*c/IA * k/dz * ( stats.norm.cdf(zrd[:,0]+0.5*dz/ppcz,scale=Ld*k,loc=z0d) - stats.norm.cdf(zrd[:,0]-0.5*dz/ppcz,scale=Ld*k,loc=z0d) ) * ( stats.weibull_min.cdf(zrd[:,1]+0.5*dr/ppcr,2,scale=2**.5*Rd*k) - stats.weibull_min.cdf(zrd[:,1]-0.5*dr/ppcr,2,scale=2**.5*Rd*k) )

else:
  Nd=1000000
  zd=stats.norm.rvs(scale=Ld*kp,loc=z0d,size=Nd)
  zd=np.sort(zd)[::-1]
  rd=stats.weibull_min.rvs(2,scale=2**.5*Rd*kp,size=Nd)
  zrd=np.c_[zd,rd]
  wd=2*Qd*c/IA / (dz/kp) / Nd * np.ones(Nd)

dummyMass=1e12
pzd=dummyMass*gammad*np.ones(Nd)
prd=np.zeros(Nd)
pad=np.zeros(Nd)
if freezd:
    qd=dummyMass**-1*np.ones(Nd)
else:
    qd=np.ones(Nd)
#2*beampop*e*c/IA / (dz/k) / N *np.ones(N)

driver=np.c_[zrd,pzd,prd,pad,qd,wd,list(range(1,Nd+1))]

# electron bunch
Ekw = 1000 # [MeV]
gammaw = Ekw / mec2 + 1
betaw = np.sqrt(1 - 1 / gammaw ** 2)
dpw = spd/100# 0.001  # relative energy spread: dp/p0
mpz=gammaw * betaw
spz=mpz*dpw
Lw=sigz
#Rw=10e-6*2**.5
z0w=z0d-xi0 #1.25-Lwindow+Lbase
emw= em #6.84e-6  #float(sys.argv[1])*1e-6    #emittance in um  #6.84e-6
Qw = qe #-1e-12*float(sys.argv[1])  #float(sys.argv[2])*-1e-12  #charge in pC     #-100e-12
freezw=0 # 1 for true; 0 for false

#Rw=8.66969*emw**.5/gammae0**.25/(n/1e6)**.25
Rw=(2*c**2*emw**2/omegap**2/gammaw)**.25
print("# Matched radius :",Rw*1e6,"um")


#number of macroparticles
Nw=1e6  #5e4  #1e6
Nw=int(Nw)

rw=stats.weibull_min.rvs(2,scale=2**.5*Rw*kp,size=Nw)
print('# std(rw):',(np.sum(rw**2)/(Nw))**.5/2**0.5*1e6/kp)
zw = stats.norm.rvs(loc=z0w, scale=Lw*kp, size=Nw)
zw=np.sort(zw)[::-1]
pzw=mpz+np.random.normal(scale=spz,size=Nw)
print(pzw)
prw=emw/Rw*np.random.normal(size=Nw)
paw=emw/Rw*np.random.normal(size=Nw)*rw
qw=-np.ones(Nw)
ww=2*Qw*c/IA / (dz/kp) / Nw * np.ones(Nw)

# print("# %d macroparticles, each corresponding to %.2f physical particles" % (Nw,ne), file=sys.stderr)
print("# sr / sr_goal",(np.sum(rw**2)/(Nw))**.5/2**0.5/Rw/kp, file=sys.stderr)
print("# sz / sz_goal",np.std(zw)/Lw/kp, file=sys.stderr)
print("# spz/spz_goal",np.std(pzw)/spz, file=sys.stderr)
print("# spr/spr_goal",np.std(prw)/(emw/Rw), file=sys.stderr)

witness=np.c_[zw,rw,pzw,prw,paw,qw,ww,list(range(Nd+Nw+1,2*Nw+1+Nd))]
endparticle=[-100000.0,0,0,0,0,1,0,0]
particles=np.r_[driver, witness,[endparticle]]
#particles=np.r_[driver,[endparticle]]
print('zds={},rds={},Nd={},zw={}'.format(zds,len(rds),Nd,len(zw)))
particles.tofile("beamfile.bin")
with open("beamfile.bit",'w') as f:
  f.write("0.0")

with open("parameters.txt", "w+") as f:
    f.write('# %s\n' % (sim_path))
    f.write('# %s\n' % (sim_name))
    f.write('\n')
    f.write('# Sim. parameters:\n')
    f.write('Nmax         = %18d\n' % t_max)
    f.write('Nbeam        = %18d\n' % save_beam)
    f.write('Nout         = %18d\n' % save_output)
    f.write('zmax         = %18.3e\n' % xi_size)
    f.write('rmax         = %18.3e\n' % r_size)
    f.write('dz           = %18.3e\n' % dz)
    f.write('dr           = %18.3e\n' % dr)
    f.write('dt           = %18.3e\n' % t_step)
    f.write('\n')
    f.write('# Plasma parameters:\n')
    f.write('n_plasma      = %18.2e\n' % n0)
    f.write('\n')
    f.write('# Proton bunch parameters:\n')
    f.write('Qp            = %18.3e\n' % (Qd))
    f.write('z0d           = %18.3e\n' % z0d)
    f.write('sig_zp        = %18.3e\n' % (Ld))
    f.write('sig_rp        = %18.3e\n' % (Rd))
    f.write('emd           = %18.3e\n' % (emd))
    f.write('zp0           = %18.3e\n' % z0d)
    f.write('gammap0       = %18.3e\n' % gammad)
    f.write('dgammad       = %18.3e\n' % dpd)
    f.write('Np            = %18.2e\n' % Nd)
    f.write('freezep       = %18d\n' % freezd)
    f.write('\n')
    f.write('# Witness bunch parameters:\n')
    f.write('Qw            = %18.3e\n' % (Qw))
    f.write('z0w           = %18.3e\n' % z0w)
    f.write('sig_zw        = %18.3e\n' % (Lw))
    f.write('sig_rw        = %18.3e\n' % (Rw))
    f.write('emw           = %18.3e\n' % (emw))
    f.write('zw0           = %18.3e\n' % z0w)
    f.write('gammaw0       = %18.3e\n' % gammaw)
    f.write('dgammaw       = %18.3e\n' % dpw)
    f.write('Nw            = %18.2e\n' % Nw)
    f.write('freezew       = %18d\n' % freezw)
    f.write('\n')
