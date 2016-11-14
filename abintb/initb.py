import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.linalg import expm

#import pylab as pl
#import sys
#import latexify as lf 
#from oneDimBerryPhase_class import BerryPhase 


class IniModel:

	def __init__(self, tau, deltau, ctrl, knum=12*6+1, dt=.1, ham_choose = 3,iniband = 'down'):
		
		self.tau = tau
		self.deltau = deltau
		self.knum = knum
		self.dt = dt
		self.ctrl = ctrl	
		self.iniband = iniband	

		self.pau_x = np.array([[0,1],[1,0]])
		self.pau_y = np.array([[0,-1j],[1j,0]])
		self.pau_z = np.array([[1,0],[0,-1]])
		self.pau_i = np.array([[1,0],[0,1]])
		
		self.tim_all = len(ctrl)
		self.dim = 2
		self.t_ini = 0.
		self.real_tim = np.array(range(self.tim_all+1)) * self.dt + self.t_ini  
		self.ham_choose = ham_choose
		self.zero = -0.001
		self.eps = np.array(range(self.knum+1))*-2*np.pi/(self.knum) + np.pi #- self.zero 
		den = abs(np.sqrt((tau+deltau)**2+(tau-deltau)**2+2*(tau**2-deltau**2)*np.cos(self.eps))/(np.sin(self.eps)-self.zero)) #not right one
		#den = np.ones(len(self.eps))
		#den = 1./abs(np.sin(self.eps)-self.zero) #not right one
		self.denFx = np.sqrt(den/sum(den))
		self.input_den = np.array(zip(self.denFx,self.denFx))
		#plt.plot( den/sum(den))
		#plt.show()

	def dvec(self, k, ctrlt):

		tau = self.tau
		deltau = self.deltau #+ ctrlt
		val = self.ham_choose 

		if val == 0:
			dx = tau+deltau + (tau-deltau) * np.cos(k-ctrlt) 
			dy = (tau-deltau) * np.sin(k-ctrlt) 
			dz = 0

		elif val == 1:
			deltau = -deltau
			dx = tau+deltau + (tau-deltau) * np.cos(k-ctrlt) 
			dy = (tau-deltau) * np.sin(k-ctrlt) 
			#dx = tau+deltau+ctrlt + (tau-deltau-ctrlt) * np.cos(k) 
			#dy = (tau-deltau-ctrlt) * np.sin(k) 
			dz = 0

		elif val == 2:
			dx = 0.#self.tau/2. 
			dy = 0.#(self.tau-deltau) * np.sin(k-A) 
			dz = tau * np.cos(k-ctrlt)
		
		elif val == 3:
			dx = self.tau/2. 
			dy = 0.#(self.tau-deltau) * np.sin(k-A) 
			#dz = (tau+ctrlt) * np.cos(k)
			dz = tau * np.cos(k-ctrlt)
		
		return dx,dy,dz	

	def ham(self, k, ctrlt):
	
		dx,dy,dz = self.dvec(k,ctrlt)
		return self.pau_x * dx + self.pau_y * dy + self.pau_z * dz
	
	def ut(self,k,ctrlt):
		
		dx,dy,dz = self.dvec(k,ctrlt)
		hami = self.ham(k,ctrlt)
		d = np.sqrt(dx**2 + dy**2 + dz**2)*self.dt
		#u = expm(-1j*hami*self.dt)
		u =  np.cos(d)*self.pau_i -1j*self.dt/d*np.sin(d)*hami
		#print np.sum(u-u)
		return u

	def phi_ini(self, k, ctrl0, state = 'mix'):
		w,v = np.linalg.eigh(self.ham(k,ctrl0))

		if state == 'mix':
			return ((v[:,0]+v[:,1])/np.sqrt(2)).reshape(2,1)

		elif state == 'down':
			return v[:,0].reshape(2,1)

		elif state == 'up':
			return v[:,1].reshape(2,1)
		else: 
			print 'no such state!!'
	
	def phi_t(self, k):

		dim = self.dim
		tim_all = self.tim_all
		ctrl = self.ctrl

		phi_ini = self.phi_ini(k,ctrl[0],self.iniband)
		phi_all = np.zeros((tim_all+1,dim,1),dtype=complex)
		phi_all[0,:,:] = phi_ini[:]
		for tim in xrange(tim_all):
			u = self.ut(k,ctrl[tim]) 
			phi_all[tim+1,:,:] = np.dot(u,phi_all[tim,:,:])
		return phi_all


	def clc_cvec(self):
		eps = self.eps
		cvec = np.array(map(self.phi_t,eps))
		return np.swapaxes(cvec[:,:,:,0],1,2)


	def eig_energy(self, k):
		ctrl0 = 0
		w, v = np.linalg.eigh(self.ham(k,ctrl0))
		return w


	def spectrum(self):
		eps = self.eps
		spec = np.array(map(self.eig_energy,eps))
		return spec


if __name__ == "__main__":

	dt = .1
	E0 = 1. 
	knum = 100 
	freq = 1. 

	tau = 1. 
	deltau = .5#-.3
	#phi_ini = [[1],[0]]

	n_tot = 4000 
	t_rel = (np.array(range(n_tot-1))-2000)*dt
	ctrl =  E0 * np.cos(freq*t_rel)

	init = IniModel(tau, deltau, ctrl, knum=knum, dt=dt)

	'''
	spec = init.spectrum()
	print spec
	plt.plot(init.eps,spec[:,0])
	plt.plot(init.eps,spec[:,1])
	plt.show()
	'''
	'''
	phi = init.phi_t(k = 0.)
	plt.plot(np.real(np.conjugate(phi)[:,1,:]*phi[:,1,:]))
	plt.plot(np.real(np.conjugate(phi)[:,0,:]*phi[:,0,:]))
	plt.show()
	'''

	ti = time()
	cvec = init.clc_cvec()
	print np.shape(cvec), time() - ti
