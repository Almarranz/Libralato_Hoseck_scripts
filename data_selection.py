#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:07:12 2022

@author: amartinez
"""
# %%imports
from astropy.table import Table

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import pandas as pd
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.table import Column
# %%plotting parametres
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 20})
rcParams.update({'figure.figsize':(10,5)})
rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.sans-serif": ["Palatino"]})
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams.update({'figure.max_open_warning': 0})# a warniing for matplot lib pop up because so many plots, this turining it of
# %%
datos = '/Users/amartinez/Desktop/PhD/Libralato_Hoseck/'
hst = Table.read(datos + 'hst_gaia_Len_F1_2j.fits')
pruebas = '/Users/amartinez/Desktop/PhD/Libralato_Hoseck/pruebas/'
gns_cat='/Users/amartinez/Desktop/PhD/Libralato_data/CATALOGS/'
# %%
# Tranforma Hoseck coordinates into Ra and Dec
hst_info = str(hst.columns)
# Positions are reported in angular de-projected units relative to the following 
# position: 
# RA_zp: 17:45:32.12
# DEC_zp: -28:56:12.66
# RA = RA_zp + (RA_0 / cos(DEC)) 
# DEC = DEC_zp + DEC_0
# Im not sure about the obsetime for this ones. Ask Mattd
center = SkyCoord(ra = '17h45m48s', dec = '-28d51m46s', unit='degree',frame = 'icrs',equinox = 'J2000', obstime = 'J2015.6')
# DEC = center.dec + hst['y'][:,1]*u.arcsec
# RA = center.ra + hst['x'][:,1]*u.arcsec/np.cos(DEC)
DEC = center.dec + hst['y0_final']*u.arcsec
RA = center.ra + hst['x0_final']*u.arcsec/np.cos(DEC)
radec = SkyCoord(ra=RA,dec=DEC, frame = 'icrs',equinox = 'J2000', obstime = 'J2015.6')

hst_coord =hst
hst_coord.add_columns([radec.ra, radec.dec], names = ['Ra','Dec'])
np.savetxt(pruebas + 'hosek_coord_all.txt',np.array([radec.ra.value,radec.dec.value]).T,fmt = '%.8f', header = 'Ra Dec') 
print(hst_coord.columns)
# %%
# Load gns1 central catalog
# _RAJ2000' #0	_DEJ2000' #1 	RAJ2000' #2	e_RAJ2000' #3	DEJ2000' #4	e_DEJ2000' #5	
# RAJdeg' #6	e_RAJdeg' #7	DEJdeg' #8	e_DEJdeg' #9	RAHdeg' #10	e_RAHdeg' #11	
# DEHdeg' #12	e_DEHdeg' #13	RAKsdeg' #14	e_RAKsdeg' #15	DEKsdeg' #16	
# e_DEKsdeg' #17	Jmag' #18	e_Jmag' #19	Hmag' #20	e_Hmag' #21	Ksmag' #22	
# e_Ksmag' #23	iJ' #24	iH #25	iKs' #26
gns_csv= pd.read_csv(gns_cat + 'GNS_central.csv')# tCentral region of GNS
gns_info = str(gns_csv.columns)
gns= gns_csv.to_numpy()
# %%
# Eliminate foreground stars via colo cut
valid = np.where((gns[:,20]<90) & (gns[:,22]<90))
gns_val = gns[valid]
color_cut = np.where((gns_val[:,20]-gns_val[:,22])>1.3)
gns_center  = gns_val[color_cut]

gns_coord = SkyCoord(ra=gns_center[:,10]*u.degree, dec=gns_center[:,12]*u.degree, frame = 'icrs', equinox = 'J2000',obstime='J2015.5')
# %%
# Match Hoseck stars to get rid of the foreground stars
max_sep = 0.5 * u.arcsec
idx,d2d,d3d = radec.match_to_catalog_sky(gns_coord,nthneighbor=1)# ,nthneighbor=1 is for 1-to-1 match
sep_constraint = d2d < max_sep
hst_center = hst[sep_constraint]
gns_match = gns_center[idx[sep_constraint]]


# %%
save_A = np.array([hst_center['Ra'],	hst_center['Dec'],	hst_center['vx_final'],	hst_center['vy_final'],	hst_center['vxe_final'],	hst_center['vye_final'],	gns_match[:,20],gns_match[:,22]]).T
np.savetxt(pruebas + 'hosek_center.txt',save_A,fmt = '%.8f', header = 'Ra Dec vx_final vy_final vxe_final vxe_final H Ks') 
# PM in galactic
pm_gal = SkyCoord(ra = hst_center['Ra'], dec = hst_center['Dec'], unit = 'degree',
                  pm_ra_cosdec = hst_center['vx_final']*u.arcsec/u.yr, pm_dec = hst_center['vy_final']*u.arcsec/u.yr,
                  frame = 'icrs', equinox = 'J2000', obstime = 'J2015.6').galactic

# %
pm_gal_ecut = np.where(np.sqrt(hst_center['vxe_final']**2 + hst_center['vye_final']**2) <0.002)
pm_gal_cut = pm_gal[pm_gal_ecut]
hst_trim = hst_center[pm_gal_ecut]
fig, ax = plt.subplots(1,2, figsize = (20,10))
bins=30
ax[0].hist(pm_gal_cut.pm_l_cosb.value, bins=bins,
           label = '$\overline{\mu_{l}}$ =%.3f\n$\sigma_{l}$ =%.3f'%(np.mean(pm_gal_cut.pm_l_cosb.value),np.std(pm_gal_cut.pm_l_cosb.value)), color = 'orange')
ax[1].hist(pm_gal_cut.pm_b.value, bins=bins,
           label = '$\overline{\mu_{b}}$ =%.3f\n$\sigma_{b}$ =%.3f'%(np.mean(pm_gal_cut.pm_b.value),np.std(pm_gal_cut.pm_b.value)), color = 'orange')
ax[0].axvline(np.mean(pm_gal_cut.pm_l_cosb.value), color ='red')
ax[1].axvline(np.mean(pm_gal_cut.pm_b.value), color ='red')
ax[0].legend()
ax[1].legend()
ax[0].set_title('Center Stars: %s. After trimming: %s'%(len(hst_center),len(pm_gal_cut.pm_l_cosb.value)))

save_B = np.array([hst_trim['Ra'],	hst_trim['Dec'],	hst_trim['vx_final'],	hst_trim['vy_final'],	hst_trim['vxe_final'],	hst_trim['vye_final']]).T
np.savetxt(pruebas + 'hosek_center_trim.txt',save_B,fmt = '%.8f', header = 'Ra Dec vx_final vy_final vxe_final vxe_final H Ks') 
# sys.exit('123')


# %%
# alpha =1
# fig, ax = plt.subplots(1,2,figsize=(20,10))
# ax[0].scatter(hst_center['F139M_ave'][pm_gal_ecut],hst_center['vxe_final'][pm_gal_ecut], alpha = alpha)
# ax[0].set_ylabel('error_vx (arscec/yr)')
# ax[0].set_xlabel('F139M')
# ax[1].scatter(hst_center['F139M_ave'][pm_gal_ecut],hst_center['vye_final'][pm_gal_ecut], alpha = alpha)
# ax[1].set_ylabel('error_vy (arscec/yr)')
# ax[1].set_xlabel('F139M')

# %%
vxy_e = np.sqrt(hst_center['vxe_final']**2 + hst_center['vye_final']**2)
ve_i_valid = []
paso = 1
vx_lim = 0.0025
perc_good = 85
mag_b=np.digitize(hst_center['F139M_ave'], np.arange(np.round(min(hst_center['F139M_ave'])),np.round(max(hst_center['F139M_ave'])+1),paso), right=True)

for i in range(min(mag_b),(max(mag_b)+1)):
    try:
        mag_binned=np.where(mag_b==i)
        ve_i=vxy_e[mag_binned]
        print('%.5f'%(np.percentile(ve_i,perc_good)),i,len(ve_i),len(mag_binned[0]))
        perc = np.percentile(ve_i,perc_good)
        for j in range(len(ve_i)):
            if ve_i[j] < vx_lim:
                if ve_i[j] <= perc or ve_i[j] <= 0.002:
                    ve_i_valid.append(mag_binned[0][j])
    except:
        print('Fallo:',i,len(ve_i),len(mag_binned[0]))
        
hst_trimB = hst_center[ve_i_valid]
H_trimm, Ks_trimm = gns_match[:,20][ve_i_valid],gns_match[:,22][ve_i_valid]
save_C = np.array([hst_trimB['Ra'],	hst_trimB['Dec'], hst_trimB['x0_final'],hst_trimB['y0_final'],
                   hst_trimB['vx_final'], hst_trimB['vy_final'],	hst_trimB['vxe_final'],	hst_trimB['vye_final'],
                   H_trimm, Ks_trimm]).T
np.savetxt(pruebas + 'hosek_center_trimB.txt',save_C,fmt = '%.8f', header = 'Ra Dec x y vx_final vy_final vxe_final vxe_final H Ks') 

# %
fig, ax = plt.subplots(1,2, figsize = (20,10))
bins=30
ax[0].hist(pm_gal.pm_l_cosb.value[ve_i_valid], bins=bins,
           label = '$\overline{\mu_{l}}$ =%.3f\n$\sigma_{l}$ =%.3f'%(np.mean(pm_gal.pm_l_cosb.value[ve_i_valid]),np.std(pm_gal.pm_l_cosb.value[ve_i_valid])))
ax[1].hist(pm_gal.pm_b.value[ve_i_valid], bins=bins,
           label = '$\overline{\mu_{b}}$ =%.3f\n$\sigma_{b}$ =%.3f'%(np.mean(pm_gal.pm_b.value[ve_i_valid]),np.std(pm_gal.pm_b.value[ve_i_valid])))
ax[0].axvline(np.mean(pm_gal.pm_l_cosb.value[ve_i_valid]), color ='red')
ax[1].axvline(np.mean(pm_gal.pm_b.value[ve_i_valid]), color ='red')
ax[0].legend()
ax[1].legend()
ax[0].set_title('Center Stars: %s. After trimming: %s'%(len(hst_center),len(ve_i_valid)))
# %%
# alpha =1
# fig, ax = plt.subplots(1,2,figsize=(20,10))
# ax[0].scatter(hst_center['F139M_ave'][ve_i_valid],hst_center['vxe_final'][ve_i_valid], alpha = alpha)
# ax[0].set_ylabel('error_vx (arscec/yr)')
# ax[0].set_xlabel('F139M')
# ax[1].scatter(hst_center['F139M_ave'][ve_i_valid],hst_center['vye_final'][ve_i_valid], alpha = alpha)
# ax[1].set_ylabel('error_vy (arscec/yr)')
# ax[1].set_xlabel('F139M')
# ax[0].set_title('Center Stars: %s. After trimming: %s'%(len(hst_center),len(ve_i_valid)))




# %%
print( hst_trimB['x0_final'])

