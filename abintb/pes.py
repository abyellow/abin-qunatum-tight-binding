import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from time import time

import Glf90_v2 as G90
#from sshIniData import SSHIniData
import qn, tb

class PES:

	def __init__(self, QnIni, phi_kall, kall, dos, tin, std=15, pau='i'):

		self.QnIni = QnIni
		self.dt = QnIni.dt

		self.c_vec1 = phi_kall
		self.c_vec2 = self.gen_cvec2(pau)

		self.dos = dos
		self.std = std
		self.t_rang = 3*std

		self.tin = tin
		self.kall = kall
		
		self.w_num = 200
		self.w_int = np.linspace(-np.pi,np.pi,self.w_num)


	def gen_cvec2(self,pau):
		
		c_vec1 = self.c_vec1	
		c_vec2 = np.zeros(np.shape(c_vec1), dtype = complex)

		if pau == 'z':
			c_vec2[:,0,:], c_vec2[:,1,:] = c_vec1[:,0,:], -c_vec1[:,1,:] 

		elif pau == 'x':
			c_vec2[:,0,:], c_vec2[:,1,:] = c_vec1[:,1,:], c_vec1[:,0,:] 

		elif pau == 'y':
			c_vec2[:,0,:], c_vec2[:,1,:] = 1j*c_vec1[:,1,:], -1j*c_vec1[:,0,:] 

		else:
			c_vec2[:] = c_vec1[:]
		return c_vec2

	def st_vec(self, tp, omega) :

		w = omega
		dt = self.dt
		std = self.std
		t_rang = self.t_rang	
		n_t = 2*int(t_rang/dt)

		time = dt*np.array(range(n_t))+tp-t_rang
		st_vec = dt*np.exp(-((time-tp)/std)**2)*np.exp(-1j*w*(time-tp))
		return st_vec/np.sum(np.abs(st_vec))


	def gen_lessG(self,c_veck1,c_veck2,denk):

		n_tot = len(c_veck1[0,0,:])
		k_tot = len(c_veck1[:,0,0])

		ts = time()	
		Gless = np.matrix(np.zeros((n_tot,n_tot),dtype=complex))
		Gless = G90.gless_v2(np.conj(c_veck1), c_veck2, denk, n_tot, k_tot)/(2*np.sum(denk[:,0]*denk[:,1]))
		print 'Gless_fortran_time:', time()-ts

		return 1j*Gless


	def int_lessG(self,Glsk,omega,tp):
		
		dt = self.dt
		t_ini = self.tin
		c_vec = self.c_vec1
		t_rang = self.t_rang

		if tp < t_ini + t_rang:
			print 'Error: Your imput initial time is %.2f so you can only start out tp time after %.2f'%(tin,tin+t_rang)
			
		else:
			n_std = int(t_rang/dt)
			n_tot = len(c_vec[0,0,:])-2*n_std
			n_tp = int((tp - t_ini)/dt)
			n_lb = n_tp - n_std
			n_ub = n_tp + n_std
			st_vec1 = np.matrix(self.st_vec(tp,omega))
			PES = (np.conj(st_vec1)*Glsk[n_lb:n_ub,n_lb:n_ub]*(st_vec1).T)[0,0]
		
		return np.imag(PES)


	
	def clc_PESk(self,Glsk,tp):

		w_int = self.w_int
		PESmtx = []
		#print 'generated PESmtx'
		for i, omega in enumerate(w_int):
			ans = self.int_lessG(Glsk,omega,tp)
			PESmtx.append(ans)
		
		return PESmtx


	def gen_PESk(self, knum, tp):

		k = knum		
		eps = self.kall
		den = self.dos 

		c_vec1 = self.c_vec1 
		c_vec2 = self.c_vec2

		denk = den[k:k+1,:] 
		c_veck1 = c_vec1[k:k+1,:,:]
		c_veck2 = c_vec2[k:k+1,:,:]

		Glsk = self.gen_lessG(c_veck1,c_veck2,denk)
		PESk = self.clc_PESk(Glsk,tp)
		print 'knum = %d,  eps = %.2f' %(k,eps[k])
		
		return PESk

		
	def gen_PESall(self, tp):
		
		ktimes = len(self.kall)
		PES2D = []
		for k in range(ktimes):
			PESk = self.gen_PESk(k,tp)
			PES2D.append(PESk)

		save_name = self.QnIni.save_name+'.txt'	
		np.savetxt(save_name,zip(*PES2D))
		
		return zip(*PES2D)

		

	def plot(self,PESmtx,ax,bar_val=True,label_val=True, color = 'RdYlBu'):

		cax = ax.imshow(PESmtx,cmap = color, vmin=-1.,vmax=1.,norm=SymLogNorm(10**-4),\
					 extent = [-3.14/2.,3.14/2.,-3.14,3.14])
		
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
		return cax



if __name__ == "__main__":

	
	dt = .1
	E0 = 1. 
	knum = 10 
	freq = 1. 

	tau = 1. 
	deltau = .5#-.3
	keps = np.linspace(-np.pi,np.pi,knum)

	n_tot = 4000 
	t_rel = (np.array(range(n_tot-1))-2000)*dt
	ctrli =  E0 * np.cos(freq*t_rel)
	cond1 = qn.QnIni(k=0.001, ctrlt=ctrli)

	ti = time()
	tb1 = tb.TbModel(cond1, keps)	
	ckall = np.array(tb1.phi_kall())
	print ckall.shape
	print 'run_time: ', time() - ti

	do = np.ones(len(keps))
	dos = np.sqrt(do/sum(do))
	dos2 = np.array(zip(dos,dos))
	tin = -200
	pes1 = PES(cond1, ckall, keps, dos2, tin,pau= 'x')
	PESmtx1 = pes1.gen_PESall(tp=0)

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	pes1.plot(PESmtx1,ax1)
	plt.show()
