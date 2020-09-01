import disort
import numpy as np

# Perform DISORT'S built-in test 1 by populating every variable
nlyr = 6
nmom = 8
nstr = 8
numu = 4
nphi = 1
ntau = 5
usrang = True
usrtau = True
ibcnd = 0
onlyfl = False
prnt = np.array([True, False, False, False, True])
plank = False
lamber = True
deltamplus = False
do_pseudo_sphere = False
dtauc = np.linspace(1, 6, endpoint=True, num=6)
ssalb = np.linspace(0.650000036, 0.900000036, endpoint=True, num=6)
pmom = np.zeros((9, 6))
pmom[0, :] = 1
temper = np.zeros(7)
wvnmlo = 0
wvnmhi = 80000
utau = np.array([0, 1.04999995, 2.09999990, 6, 21])
umu0 = 0.5
phi0 = 0
umu = np.array([-1, -0.200000003, 0.200000003, 1])
phi = np.array([1])
fbeam = 0
fisot = 0.318309873
albedo = 0
btemp = 320
ttemp = 100
temis = 1
earth_radius = 6371
h_lyr = np.zeros(7)
rhoq = np.zeros((4, 5, 8))
rhou = np.zeros((4, 5, 8))
rho_accurate = np.zeros((4, 1))
bemst = np.zeros(4)
emust = np.zeros(4)
accur = 0
header = 'Test Case No. 9a:  Ref. DGIS, Tables VI-VII, beta=l=0 (multiple inhomogeneous layers)'
rfldir = np.zeros(5)
rfldn = np.zeros(5)
flup = np.zeros(5)
dfdt = np.zeros(5)
uavg = np.zeros(5)
uu = np.zeros((4, 5, 1))
albmed = np.zeros(4)
trnmed = np.zeros(4)

# If desired, print the module documentation which shows all functions
#print(disort1.__doc__)

# More practically, if desired print the "disort" function's inputs and outputs
print(disort.disort.__doc__)

# Run disort
dfdt, uavg, uu = disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber, deltamplus, do_pseudo_sphere, dtauc,
                               ssalb, pmom, temper, wvnmlo, wvnmhi, utau, umu0, phi0, umu, phi, fbeam, fisot, albedo,
                               btemp, ttemp, temis, earth_radius, h_lyr, rhoq, rhou, rho_accurate, bemst, emust, accur,
                               header, rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed, maxcly=nlyr, maxmom=nmom,
                               maxcmu=nstr, maxumu=numu, maxphi=nphi, maxulv=ntau)

print(dfdt)
print(uavg)
print(uu)
