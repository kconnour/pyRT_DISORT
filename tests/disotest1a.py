'''import disort
import numpy as np

# Perform DISORT'S built-in test 1 by populating every variable
nlyr = 1
nmom = 16
nstr = 16
numu = 6
nphi = 1
ntau = 2
usrang = True
usrtau = True
ibcnd = 0
onlyfl = False
prnt = np.array([True, False, False, False, True])
plank = False
lamber = True
deltamplus = False
do_pseudo_sphere = False
dtauc = 3.125*10**-2
ssalb = 0.200000003
pmom = np.zeros(17)
pmom[0] = 1
temper = np.zeros(2)
wvnmlo = 0
wvnmhi = 0
utau = np.array([0, 3.125*10**-2])
umu0 = 0.100000001
phi0 = 0
umu = np.array([-1, -0.5, -0.100000001, 0.100000001, 0.5, 1])
phi = np.array([0])
fbeam = 31.4159279
fisot = 0
albedo = 0
btemp = 0
ttemp = 0
temis = 0
earth_radius = 6371
h_lyr = np.zeros(2)
rhoq = np.zeros((8, 9, 16))
rhou = np.zeros((6, 9, 16))
rho_accurate = np.zeros((6, 1))
bemst = np.zeros(8)
emust = np.zeros(6)
accur = 0
header = 'Test Case No. 1a:  Isotropic Scattering, Ref. VH1, Table 12:  b =  0.03125, a = 0.20'
rfldir = np.zeros(2)
rfldn = np.zeros(2)
flup = np.zeros(2)
dfdt = np.zeros(2)
uavg = np.zeros(2)
uu = np.zeros((6, 2, 1), 'f')
albmed = np.zeros(6)
trnmed = np.zeros(6)

# If desired, print the module documentation which shows all functions
#print(disort1.__doc__)

# More practically, if desired print the "disort" function's inputs and outputs
print(disort.disort.__doc__)

# Run disort, putting DFDT, UAVG, and UU in a, b, and c, respectively
rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed = disort.disort(usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber, deltamplus, do_pseudo_sphere, dtauc, ssalb,
                        pmom, temper, wvnmlo, wvnmhi, utau, umu0, phi0, umu, phi, fbeam, fisot, albedo, btemp, ttemp,
                        temis, earth_radius, h_lyr, rhoq, rhou, rho_accurate, bemst, emust, accur, header, rfldir,
                        rfldn, flup, dfdt, uavg, uu, albmed, trnmed)

print(uu)

# If I put in the inputs in the same order as the example, it fails
output = disort1.disort(nlyr, nmom, nstr, numu, nphi, ntau, usrang, usrtau, ibcnd, onlyfl, prnt, plank, lamber, deltamplus,
                        do_pseudo_sphere, dtauc, ssalb, pmom, temper, wvnmlo, wvnmhi, utau, umu0, phi0, umu, phi,
                        fbeam, fisot, albedo, btemp, ttemp, temis, earth_radius, h_lyr, rhoq, rhou, rho_accurate, bemst,
                        emust, accur, header, rfldir, rfldn, flup, dfdt, uavg, uu, albmed, trnmed)'''
