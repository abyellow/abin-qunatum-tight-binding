import sys
from time import time
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

import Glf90_v2 as G90
#from lessG_class import lessG
from sshIniData import SSHIniData

class PES:

	def __init__(self, iniData, c_vec, tin, E0, freq, dtp=1., std=10, width = 1):

		self.iniData = iniData
		self.dt = self.iniData.dt
		self.ktimes= self.iniData.knum
		self.tau = self.iniData.tau
		self.deltau = self.iniData.deltau
		self.ham_choose = self.iniData.ham_choose
		self.nearZero = self.iniData.zero
		self.input_den = self.iniData.input_den 
		self.epsFx =  self.iniData.eps
		self.width = width
		self.c_vec = c_vec
		self.std = std
		self.t_rang = 3*std
		self.tin = tin
		self.dtp = dtp
		self.w_num = 2*self.ktimes
		self.w_int = np.array(range(self.w_num+1))*-2*(np.pi+self.nearZero)/(self.w_num)+np.pi
		self.E0 = E0
		self.freq = freq
		self.iniband = self.iniData.iniband

	def st_vec(self, tp, w) :

		dt = self.dt
		std = self.std
		t_rang = self.t_rang	
		n_t = 2*int(t_rang/dt)

		time = dt*np.array(range(n_t))+tp-t_rang
		st_vec = dt*np.exp(-((time-tp)/std)**2)*np.exp(-1j*w*(time-tp))
		#plt.plot(time,st_vec/np.sum(np.abs(st_vec)*dt))
		#plt.show()
		return st_vec/np.sum(np.abs(st_vec))#*np.sqrt(1.42))#(len(st_vec)*dt/np.pi)#(sum(abs(st_vec)))


	def gen_lessG(self,c_veck1,c_veck2,input_denk):

		n_tot = len(c_veck1[0,0,:])
		k_tot = len(c_veck1[:,0,0])

		ts = time()	
		Gless = np.matrix(np.zeros((n_tot,n_tot),dtype=complex))
		#print 'cvec size for Gless: (where should be wrong!)',np.sum(c_veck2-np.conj(c_veck2)), np.shape(c_veck2.T)
		Gless = G90.gless_v2(np.conj(c_veck1), c_veck2, input_denk, n_tot, k_tot)/(2*np.sum(input_denk[:,0]*input_denk[:,1]))
		print 'Gless_fortran_time:', time()-ts

		return 1j*Gless


	def int_PES(self,Glsk,omega,tp):

		std = self.std
		dt = self.dt
		t_ini = self.tin	
		c_vec = self.c_vec
		t_rang = self.t_rang
		dtp = self.dtp

		n_std = int(t_rang/dt)
		n_tot = len(c_vec[0,0,:])-2*n_std
		n_tp = int((tp - t_ini)/dt)
		n_lb = n_tp - n_std
		n_ub = n_tp + n_std
		st_vec1 = np.matrix(self.st_vec(tp,omega))
		PES = (np.conj(st_vec1)*Glsk[n_lb:n_ub,n_lb:n_ub]*(st_vec1).T)[0,0]
		#print PES, np.shape(PES)
		#PES = (np.conj(st_vec1)*(st_vec1).T)[0,0]
		
		return np.imag(PES)


	
	def clc_PES(self,Glsk,tp):

		w_int = -self.w_int
		width = self.width
		PESmtx = []
		#print 'generated PESmtx'
		for i, omega in enumerate(w_int):
			ans = self.int_PES(Glsk,omega,tp)
			for j in range(width):
				PESmtx.append(ans)
		
		return PESmtx


	def final_run(self, tp, pau = 'i'):

		epsFx = self.epsFx
		dt = self.dt
		ktimes = self.ktimes
		deltau = self.deltau
		ham_choose = self.ham_choose
		iniband = self.iniband
		std = self.std
		E0 = self.E0
		freq = self.freq
		input_den = self.input_den 

		c_vec1 = self.c_vec 
		c_vec2 = np.zeros(np.shape(c_vec1), dtype = complex)

		if pau == 'z':
			c_vec2[:,0,:], c_vec2[:,1,:] = c_vec1[:,0,:], -c_vec1[:,1,:] 

		elif pau == 'x':
			c_vec2[:,0,:], c_vec2[:,1,:] = c_vec1[:,1,:], c_vec1[:,0,:] 

		elif pau == 'y':
			c_vec2[:,0,:], c_vec2[:,1,:] = 1j*c_vec1[:,1,:], -1j*c_vec1[:,0,:] 

		else:
			c_vec2[:] = c_vec1[:]

		#print 'cvec1-cvec2: ', np.sum(c_vec1[:,1,:] - c_vec2[:,1,:])
		PES2D = []
		for k in range(ktimes):
			denk = input_den[k:k+1,:] 
			c_veck1 = c_vec1[k:k+1,:,:]
			c_veck2 = c_vec2[k:k+1,:,:]

			Gls = self.gen_lessG(c_veck1,c_veck2,denk)
			#n_tot = len(c_veck1[0,0,:])
			#Gls = 1j*np.matrix(np.ones((n_tot,n_tot),dtype=complex))
			#print Gls
			PESmtx = self.clc_PES(Gls,tp)
			PES2D.append(PESmtx)
			print 'k = %d,  eps = %.2f' %(k,epsFx[k])
			
		save_name = 'data/PES2ssh_ham_%d_dt_%.2f_ktimes_%d_tp_%.1f_E0_%.1f_freq_%.2f_deltau_%.1f_paui_%s_std_%.1f_band_%s.txt' %(ham_choose, dt,ktimes,tp,E0,freq,deltau,pau,std,iniband)
		np.savetxt(save_name,zip(*PES2D))
		
		return zip(*PES2D)

		

	def plot(self,PESmtx,ax,bar_val=True,label_val=True, color = 'RdYlBu'):


		cax = ax.imshow(PESmtx,cmap = color, vmin=-1.,vmax=1.,norm=SymLogNorm(10**-4), extent = [-3.14/2.,3.14/2.,-3.14,3.14])
		
		fsize = 16
		plt.xticks([-1.5,-.75,0.0,.75,1.5], ['-1.0','-0.5','0.0','0.5','1.0'],fontsize = fsize)
		plt.yticks([-3,-2,-1,0,1,2,3], ['-3','-2','-1','0','1','2','3'],fontsize = fsize)
		if label_val:
			xlab = r' $k ( a/ \pi )$'
			ylab = r'Energy / $\tau$'
			plt.xlabel(xlab, fontsize = fsize)
			plt.ylabel(ylab, fontsize = fsize)

		if bar_val:
			tick_labels=['-0.500','-0.005','0.000','0.005','0.500']
			tick_locs=[-0.50,-0.005,0.00,0.005,0.50]
			bar = plt.colorbar(cax,ticks=tick_locs,shrink=0.6)
			bar.set_ticklabels(tick_labels)
			bar.update_ticks()
		#lf.latexify()
		return cax



#if __name__ == "__main__":

	
