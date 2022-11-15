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
import astropy.coordinates as ap_coor
import pandas as pd
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter

import math
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
# Here we trimmed the data making juts a cut in the pms
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
# Here the data is trimming by magnitudes bins
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
pm = np.array([hst_trimB['Ra'],	hst_trimB['Dec'], hst_trimB['x0_final'],hst_trimB['y0_final'],
                   hst_trimB['vx_final'], hst_trimB['vy_final'],	hst_trimB['vxe_final'],	hst_trimB['vye_final'],
                   pm_gal.l.value[ve_i_valid],
                   pm_gal.b.value[ve_i_valid],
                   pm_gal.pm_l_cosb.value[ve_i_valid],
                   pm_gal.pm_b.value[ve_i_valid],
                   H_trimm, Ks_trimm]).T
np.savetxt(pruebas + 'hosek_center_trimB.txt',pm,fmt = '%.8f', header = 'Ra Dec x y vx_final vy_final vxe_final vxe_final l b mul mub H Ks') 

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
alpha =1
fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].scatter(hst_center['F139M_ave'][pm_gal_ecut],hst_center['vxe_final'][pm_gal_ecut], alpha = alpha)
ax[0].set_ylabel('error_vx (arscec/yr)')
ax[0].set_xlabel('F139M')
ax[1].scatter(hst_center['F139M_ave'][pm_gal_ecut],hst_center['vye_final'][pm_gal_ecut], alpha = alpha)
ax[1].set_ylabel('error_vy (arscec/yr)')
ax[1].set_xlabel('F139M')
# %%
clust_cand = np.loadtxt('/Users/amartinez/Desktop/PhD/Libralato_data/good_clusters/Sec_B_dmu1_at_minimun_2022-08-30_cluster_num32_2_knn10_area7.49/cluster32_0_0_knn_10_area_7.49_all_color.txt')
cand_coord = SkyCoord(ra = clust_cand[:,0]*u.deg, dec = clust_cand[:,1]*u.deg, frame ='icrs', equinox = 'J2000', obstime = 'J2015.5')

color = pd.read_csv('/Users/amartinez/Desktop/PhD/python/colors_html.csv')
strin= color.values.tolist()
indices = np.arange(0,len(strin),1)
# divis_list = [1,2,3]#TODO
divis_list = [1,2]#TODO

samples_list =[10,9,8,7,6,5]#TODO
# samples_list =[8]#TODO
sim_lim = 'mean'#TODO options: minimun or mean
gen_sim ='Kernnel'#TODO it is not yet implemented shuffle
clustered_by = 'pm_color'#Reminiscent of a previous script
cluster_by = 'pm'
clus_num = 0

m = (18-108)/(-100.3-1)#Points of a parallel line to the edge of the data
m1 = (-88 - 7)/(-23--102)
lim_pos_up, lim_pos_down = 18-m*(-100.3), -65 #intersection of the positives slopes lines with y axis,
lim_neg_up, lim_neg_down =110,-88-m1*(-23) #intersection of the negayives slopes lines with y axis,

# distancia entre yg_up e yg_down
dist_pos = abs(lim_pos_down-lim_pos_up)/np.sqrt((m)**2+(-1)**2)
dist_neg = abs(lim_neg_down-lim_neg_up)/np.sqrt((m1)**2+(-1)**2)
ang = math.degrees(np.arctan(m))
ang1 = math.degrees(np.arctan(m1))
sys.exit()
#   0  1  2 3   4          5        6         7     8 9 10  11 12 13
# 'Ra Dec x y vx_final vy_final vxe_final vxe_final l b mul mub H Ks'
for div in range(len(divis_list)):
    divis = divis_list[div]
    xg = np.linspace(min(pm[:,2]),max(pm[:,2]),(divis+1)*2-1)
    yg = np.linspace(min(pm[:,3]),max(pm[:,3]),(divis+1)*2-1)
    for sd in range(len(samples_list)):
        samples_dist = samples_list[sd]
        for xi in range(len(xg)-2):
            for yi in range(len(xg)-2):
                fig, ax = plt.subplots(1,1)
                ax.scatter(pm[:,2],pm[:,3],color ='k',alpha=0.1)
                valid = np.where((pm[:,2] < xg[xi+2])
                                 & (pm[:,2] >xg[xi]) 
                                 & (pm[:,3] < yg[yi+2]) 
                                 & (pm[:,3] >yg[yi]))
                pm_sub = pm[valid]
                ax.scatter(pm_sub[:,2],pm_sub[:,3],s =5,color=strin[np.random.choice(indices)] )
                coordenadas = SkyCoord(ra=pm_sub[:,0]*u.degree, dec=pm_sub[:,1]*u.degree,frame ='icrs', equinox = 'J2000', obstime = 'J2015.6')#
                mul,mub = pm_sub[:,8],pm_sub[:,9]
                x,y = pm_sub[:,2], pm_sub[:,3]
                colorines = pm_sub[:,10]-pm_sub[:,11]
                H_datos, K_datos = pm_sub[:,10], pm_sub[:,11]
                
                area = (xg[xi+1]- xg[xi])*(yg[yi+1]- yg[yi])*0.05**2/3600
                mul_kernel, mub_kernel = gaussian_kde(mul), gaussian_kde(mub)
                x_kernel, y_kernel = gaussian_kde(x), gaussian_kde(y)
                color_kernel = gaussian_kde(colorines)
        
                X=np.array([mul,mub,x,y,colorines]).T
                X_stad = StandardScaler().fit_transform(X)
                tree = KDTree(X_stad, leaf_size=2) 
                    
               
                dist, ind = tree.query(X_stad, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                d_KNN=sorted(dist[:,-1])#distance to the Kth neighbour
                lst_d_KNN_sim = []
                for d in range(5):
                    mub_sim,  mul_sim = mub_kernel.resample(len(pm_sub)), mul_kernel.resample(len(pm_sub))
                    x_sim, y_sim = x_kernel.resample(len(pm_sub)), y_kernel.resample(len(pm_sub))
                    color_sim = color_kernel.resample(len(pm_sub))
                    X_sim=np.array([mul_sim[0],mub_sim[0],x_sim[0],y_sim[0],color_sim[0]]).T
                    X_stad_sim = StandardScaler().fit_transform(X_sim)
                    tree_sim =  KDTree(X_stad_sim, leaf_size=2)
                    
                    dist_sim, ind_sim = tree_sim.query(X_stad_sim, k=samples_dist) #DistNnce to the 1,2,3...k neighbour
                    d_KNN_sim=sorted(dist_sim[:,-1])#distance to the Kth neighbour
                    
                    lst_d_KNN_sim.append(min(d_KNN_sim))
                d_KNN_sim_av = np.mean(lst_d_KNN_sim)
                fig, ax = plt.subplots(1,1,figsize=(10,10))
                ax.hist(d_KNN,bins ='auto',histtype ='step',color = 'k')
                ax.hist(d_KNN_sim,bins ='auto',histtype ='step',color = 'r')
                ax.set_xlabel('%s-NN distance'%(samples_dist)) 
                
                if sim_lim == 'mean':
                    eps_av = round((min(d_KNN)+d_KNN_sim_av)/2,3)
                    valor = d_KNN_sim_av
                elif sim_lim == 'minimun':
                    eps_av = round((min(d_KNN)+min(lst_d_KNN_sim))/2,3)
                    valor = min(lst_d_KNN_sim)
                texto = '\n'.join(('min real d_KNN = %s'%(round(min(d_KNN),3)),
                                'min sim d_KNN =%s'%(round(valor,3)),
                                'average = %s'%(eps_av),'%s'%(sim_lim),'%s'%(gen_sim)))
                
                props = dict(boxstyle='round', facecolor='w', alpha=0.5)
                # place a text box in upper left in axes coords
                ax.text(0.65, 0.25, texto, transform=ax.transAxes, fontsize=20,
                    verticalalignment='top', bbox=props)
                
                ax.set_ylabel('N') 
                
                clustering = DBSCAN(eps=eps_av, min_samples=samples_dist).fit(X_stad)
                l=clustering.labels_
                
                n_clusters = len(set(l)) - (1 if -1 in l else 0)
                # print('Group %s.Number of cluster, eps=%s and min_sambles=%s: %s'%(group,round(epsilon,2),samples,n_clusters))
                n_noise=list(l).count(-1)
                # %
                u_labels = set(l)
                colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(set(l)))]# Returns a color for each cluster. Each color consists in four number, RGBA, red, green, blue and alpha. Full opacity black would be then 0,0,0,1
                # %
                
                # %
                for k in range(len(colors)): #give noise color black with opacity 0.1
                    if list(u_labels)[k] == -1:
                        colors[k]=[0,0,0,0.1]
                # %      
                colores_index=[]
                
                for c in u_labels:
                    cl_color=np.where(l==c)
                    colores_index.append(cl_color)

                for i in range(len(set(l))-1):
                    # ax[0].scatter(pm_sub[:,2][colores_index[i]], pm_sub[:,3][colores_index[i]], color = 'orange')
                    c2 = SkyCoord(ra = pm_sub[:,0][colores_index[i]],dec = pm_sub[:,1][colores_index[i]], unit ='degree',  equinox = 'J2000', obstime = 'J2015.4')
                    fig, ax = plt.subplots(1,3,figsize=(30,10))
                    color_de_cluster = 'lime'
                    # fig, ax = plt.subplots(1,3,figsize=(30,10))
                    # ax[2].invert_yaxis()
                   
                    ax[0].set_title('Min %s-NN= %s. cluster by: %s '%(samples_dist,round(min(d_KNN),3),clustered_by))
                    # t_gal['l'] = t_gal['l'].wrap_at('180d')
                    ax[0].scatter(X[:,0][colores_index[-1]],X[:,1][colores_index[-1]], color=colors[-1],s=50,zorder=1)
                    ax[0].scatter(X[:,0],X[:,1], color=colors[-1],s=50,zorder=1)
                    # ax[1].quiver(t_gal['l'][colores_index[-1]].value,t_gal['b'][colores_index[-1]].value, X[:,0][colores_index[-1]]-pms[2], X[:,1][colores_index[-1]]-pms[3], alpha=0.5, color=colors[-1])
                
                    ax[0].scatter(X[:,0][colores_index[i]],X[:,1][colores_index[i]], color=color_de_cluster ,s=50,zorder=3)
                    # ax[0].set_xlim(-10,10)
                    # ax[0].set_ylim(-10,10)
                    ax[0].set_xlabel(r'$\mathrm{\mu_{l} (mas\ yr^{-1})}$',fontsize =30) 
                    ax[0].set_ylabel(r'$\mathrm{\mu_{b} (mas\ yr^{-1})}$',fontsize =30) 
                    ax[0].invert_xaxis()
                    
                    mul_sig, mub_sig = np.std(X[:,0][colores_index[i]]), np.std(X[:,1][colores_index[i]])
                    mul_mean, mub_mean = np.mean(X[:,0][colores_index[i]]), np.mean(X[:,1][colores_index[i]])
                    
                    mul_sig_all, mub_sig_all = np.std(X[:,0]), np.std(X[:,1])
                    mul_mean_all, mub_mean_all = np.mean(X[:,0]), np.mean(X[:,1])
                     
                    ax[0].axvline(mul_mean_all, color ='red')
                    ax[0].axhline(mub_mean_all, color ='red')
                    
                    ax[0].axvline(mul_mean_all + mul_sig_all,linestyle = 'dashed', color ='red')
                    ax[0].axvline(mul_mean_all - mul_sig_all,linestyle = 'dashed', color ='red')
                    
                    ax[0].axhline(mub_mean_all + mub_sig_all,linestyle = 'dashed', color ='red')
                    ax[0].axhline(mub_mean_all - mub_sig_all,linestyle = 'dashed', color ='red')
                
                    vel_txt = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean,3), round(mub_mean,3)),
                                         '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig,3), round(mub_sig,3)))) 
                    vel_txt_all = '\n'.join(('mul = %s, mub = %s'%(round(mul_mean_all,3), round(mub_mean_all,3)),
                                         '$\sigma_{mul}$ = %s, $\sigma_{mub}$ = %s'%(round(mul_sig_all,3), round(mub_sig_all,3))))
                    
                    propiedades = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
                    propiedades_all = dict(boxstyle='round', facecolor=colors[-1], alpha=0.1)
                    ax[0].text(0.05, 0.95, vel_txt, transform=ax[0].transAxes, fontsize=30,
                        verticalalignment='top', bbox=propiedades)
                    ax[0].text(0.05, 0.15, vel_txt_all, transform=ax[0].transAxes, fontsize=20,
                        verticalalignment='top', bbox=propiedades_all)   
                    #This calcualte the maximun distance between cluster members to have a stimation of the cluster radio
                    sep = [max(c2[c_mem].separation(c2)) for c_mem in range(len(c2))]
                    rad = max(sep)/2
                    
                    m_point = SkyCoord(ra =[np.mean(c2.ra)], dec = [np.mean(c2.dec)],frame ='icrs', equinox = 'J2000', obstime = 'J2022.4')             
                    idxc, group_md, d2d,d3d =  ap_coor.search_around_sky(m_point,coordenadas, rad*2)
                    
                    ax[0].scatter(mul[group_md],mub[group_md], color='red',s=50,zorder=1,marker='x',alpha = 0.7)
                
                    prop = dict(boxstyle='round', facecolor=color_de_cluster , alpha=0.2)
                    ax[1].text(0.15, 0.95, 'aprox cluster radio = %s"\n cluster stars = %s '%(round(rad.to(u.arcsec).value,2),len(colores_index[i][0])), transform=ax[1].transAxes, fontsize=30,
                                            verticalalignment='top', bbox=prop)
                    # ax[1].scatter(MS_coord.ra, MS_coord.dec, s=20, color ='b', marker ='.')
                    ax[1].scatter(pm_sub[:,0], pm_sub[:,1], color='k',s=50,zorder=1,alpha=0.1)#
                    # ax[1].scatter(datos[:,5],datos[:,6],color='k' ,s=50,zorder=1,alpha=0.01)
                    ax[1].scatter(pm_sub[:,0][colores_index[i]],pm_sub[:,1][colores_index[i]],color=color_de_cluster ,s=50,zorder=3)
                
                    ax[1].scatter(pm_sub[:,0][group_md],pm_sub[:,1][group_md],s=50,color='r',alpha =0.1,marker ='x')
                    ax[1].set_xlabel('Ra(deg)',fontsize =30) 
                    ax[1].set_ylabel('Dec(deg)',fontsize =30) 
                    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax[1].set_title('Area_%s_%s =%.2f $arcmin^{2}$'%(xi,yi,area))
                    
                    for cand_star in range(len(cand_coord)):
                        sep_cand = c2.separation(cand_coord[cand_star])
                        if min(sep_cand.value) <1/3600:
                            print('SOMETHING is CLOSE!')
                            ax[1].scatter(cand_coord.ra, cand_coord.dec,color ='b',s =80,marker ='x',zorder =3)
                            ax[0].set_facecolor('lavender')
                            ax[1].set_facecolor('lavender')
                            ax[2].set_facecolor('lavender')
# %%























