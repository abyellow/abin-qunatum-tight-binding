import numpy as np
from scipy.linalg import expm

import matplotlib.pyplot as plt
from time import time


class QnModel: 

	"""
	Initial data/conditions of Quantum Hamiltonian and initial states.
	"""
	def __init__(self, QnIni, tb_model=False):
	
		self.QnIni = QnIni
		self.k = QnIni.k 
		self.ctrlt = QnIni.ctrlt #np.array(ctrlt)  #initial control/laser
		self.H0 = QnIni.H0          #Hamiltonian with no control/laser
		self.Hctrl = QnIni.ham_t()#np.array(Hctrlt)    #Hamiltonian of control/laser term
		self.phi_i = QnIni.phi_i()    #initial quantum states
		self.dt = QnIni.dt          #time step size

		self.tb_model = tb_model 
		self.dim = np.shape(self.H0)[0]  #dimension of Hamiltonian
		self.t_ini = 0.       #start time

		self.tim_all = np.shape(self.Hctrl)[0] #time length of ctrl/laser
		self.real_tim = np.array(range(self.tim_all+1)) * self.dt +\
				 self.t_ini	       #real time of time length 

		self.pau_x = np.array([[0,1],[1,0]])
		self.pau_y = np.array([[0,-1j],[1j,0]])
		self.pau_z = np.array([[1,0],[0,-1]])
		self.pau_i = np.array([[1,0],[0,1]])
		

	def u_dt(self, H, tim):
		"""propagator of dt time"""
		if self.tb_model:
			#cond = QnIni(k=self.k,ctrlt=self.ctrlt)
			dx,dy,dz = self.QnIni.dvec(self.ctrlt[tim]) 
			#dx = np.real(np.trace(np.dot(self.pau_x,H))/2.)#self.dvec(k,ctrlt)
			#dy = np.real(np.trace(np.dot(self.pau_y,H))/2.)#self.dvec(k,ctrlt)
			#dz = np.real(np.trace(np.dot(self.pau_z,H))/2.)
			d = np.sqrt(dx**2 + dy**2 + dz**2)*self.dt
			u =  np.cos(d)*self.pau_i -1j*self.dt/d*np.sin(d)*H

		else:
			u =  expm(-1j*H*self.dt)

		return u


	def u_t(self):
		"""Evolve propergator for given time period"""
		dim = self.dim 
		tim_all = self.tim_all
		#ctrl = self.ctrl
		H0 = self.H0
		Hctrl = self.Hctrl

		u_all = np.zeros((tim_all+1,dim,dim),dtype = complex)
		u_all[0,:,:] = np.eye(dim)

		for tim in xrange(tim_all):
			H = H0 + Hctrl[tim]#np.matrix( ctrl[i,tim] * np.array(Hctrl[i]))
			u_all[tim+1,:,:] = np.dot(self.u_dt(H,tim), u_all[tim,:,:])


		return u_all

	def phi_t(self):
		"""Evolve state for given time period"""
		dim = self.dim
		tim_all = self.tim_all 
		phi_all = np.zeros((tim_all+1,dim,1),dtype = complex)
		phi_all[0,:,:] = self.phi_i[:]
		u_all = self.u_t()

		for tim in xrange(tim_all):
			phi_all[tim+1,:,:] = np.dot(u_all[tim+1,:,:], phi_all[0,:,:])
		
		return phi_all

	def prob_t(self,phi):
		"""probability in time"""
		return np.real(phi*np.conjugate(phi))







class QnIni:

	def __init__(self, k, ctrlt, dt = .1, tau=1.,deltau=.5, ham_val=0):

		self.dt = dt	
		self.k = k
		self.ctrlt = np.array(ctrlt)
		self.ham_val = ham_val
		self.tau = tau
		self.deltau = deltau

		self.pau_x = np.array([[0,1],[1,0]])
		self.pau_y = np.array([[0,-1j],[1j,0]])
		self.pau_z = np.array([[1,0],[0,-1]])
		self.pau_i = np.array([[1,0],[0,1]])

		self.H0 = np.zeros((2,2))


	def dvec(self,ctrl):

		k = self.k
		tau = self.tau
		deltau = self.deltau #+ ctrlt
		val = self.ham_val 

		if val == 0:
			dx = tau+deltau + (tau-deltau) * np.cos(k-ctrl) 
			dy = (tau-deltau) * np.sin(k-ctrl) 
			dz = 0

		elif val == 1:
			deltau = -deltau
			dx = tau+deltau + (tau-deltau) * np.cos(k-ctrl) 
			dy = (tau-deltau) * np.sin(k-ctrl) 
			#dx = tau+deltau+ctrlt + (tau-deltau-ctrlt) * np.cos(k) 
			#dy = (tau-deltau-ctrlt) * np.sin(k) 
			dz = 0

		elif val == 2:
			dx = 0.#self.tau/2. 
			dy = 0.#(self.tau-deltau) * np.sin(k-A) 
			dz = tau * np.cos(k-ctrl)
		
		elif val == 3:
			dx = self.tau/2. 
			dy = 0.#(self.tau-deltau) * np.sin(k-A) 
			#dz = (tau+ctrlt) * np.cos(k)
			dz = tau * np.cos(k-ctrl)

		elif val == 4:
			dx = tau+deltau + (tau-deltau) * np.cos(k[0]-ctrl) 
			dy = (tau-deltau) * np.sin(k[1]-ctrl) 
			dz = 0

	
		
		return dx,dy,dz	


	def ham(self,ctrl):

		dx,dy,dz = self.dvec(ctrl)
		return self.pau_x * dx + self.pau_y * dy + self.pau_z * dz
	
	def ham_t(self):
		ctrlt = self.ctrlt
		return np.array(map(self.ham,ctrlt))

	def phi_i(self, state = 'down'):

		w,v = np.linalg.eigh(self.ham(ctrl=0))

		if state == 'mix':
			return ((v[:,0]+v[:,1])/np.sqrt(2)).reshape(2,1)

		elif state == 'down':
			return v[:,0].reshape(2,1)

		elif state == 'up':
			return v[:,1].reshape(2,1)
		else: 
			print 'no such state!!'
	

	def eig_energy(self,ctrl=0):
		w, v = np.linalg.eigh(self.ham(ctrl))
		return w






if __name__ == "__main__":

	dt = .01
	E0 = 1. 
	knum = 100 
	freq = 1. 

	tau = 1. 
	deltau = .5#-.3
	#phi_ini = [[1],[0]]

	n_tot = 4000 
	t_rel = (np.array(range(n_tot-1))-2000)*dt
	ctrli =  E0 * np.cos(freq*t_rel)


	ki =[ 0.001	,np.pi]
	#ki = 0.001	
	ti = time()
	cond1 = QnIni(k=ki, ctrlt=ctrli,ham_val = 4)
	#phi_i = cond1.phi_i()
	#Hctrl = cond1.ham_t()
	#print Hctrl.shape
	#H0 = np.zeros((2,2))

	model1 = QnModel(cond1)
	phit = model1.phi_t()
	probt = model1.prob_t(phit)
	print 'run_time: ', time() - ti

	plt.plot(t_rel,probt[:-1,0,:])
	plt.plot(t_rel,probt[:-1,1,:])
	plt.show()

