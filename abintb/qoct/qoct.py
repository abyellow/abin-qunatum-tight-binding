"""
 Editor Bin H.
 Quantum Optimal Control Example
 One Control Parameter Model
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from time import clock

class QH:
	"""
	Initial data/conditions of Quantum Hamiltonian and initial states.
	"""
	def __init__(self, H0, Hctrl, ctrl_i, phi_i, dt=.01):

		self.H0 = H0          #Hamiltonian with no control/laser
		self.Hctrl = Hctrl    #Hamiltonian of control/laser term
		self.ctrl = ctrl_i  #initial control/laser
		self.phi_i = phi_i    #initial quantum states
		self.dt = dt          #time step size
		self.t_ini = 0.       #start time

		self.dim = np.shape(self.H0)[0]  #dimension of Hamiltonian
		self.tim_all = np.shape(self.ctrl)[0] #time length of ctrl/laser
		self.real_tim = np.array(range(self.tim_all+1)) * self.dt +\
				 self.t_ini	#real time of time length 
	
	def u_dt(self, H):
		"""propagator of dt time"""
		return  expm(-1j*H*self.dt)

	def u_next(self,H,u_now):
		"""Derive U at next time step"""
		return np.dot(self.u_dt(H), u_now)

	def u_t(self):
		"""Evolve propergator for given time period"""
		dim = self.dim 
		tim_all = self.tim_all
		ctrl = self.ctrl
		H0 = self.H0
		Hctrl = self.Hctrl

		u_all = np.zeros((tim_all+1,dim,dim),dtype = complex)
		u_all[0,:,:] = np.eye(dim)

		for tim in xrange(tim_all):
			H = H0 + np.matrix( ctrl[tim] * np.array(Hctrl) )
			u_all[tim+1,:,:] = self.u_next(H, u_all[tim,:,:])

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
		return np.real(phi*np.conjugate(phi))
	

class QOCT:
	"""
	Quantum optimal control codes
	"""
	def __init__(self, qh_input, phi_go):
		
		self.qh_in = qh_input   # class QH for all i.c. and EoM
		self.phi_go = phi_go      # goal quantum states we expect
		self.lmda = 10.         # learning rate
		self.iter_time = 1000
		self.error_bd = 10**-4  # error bound of convergence 

	
	def u_prev(self, H, u_now):
		"""Derive U at next time step"""
		return np.dot(u_now, self.qh_in.u_dt(H))

	def u_t_back(self):
		"""Evolve propergator backward for given time period"""
		
		dim = self.qh_in.dim 
		tim_all = self.qh_in.tim_all
		ctrl = self.qh_in.ctrl
		H0 = self.qh_in.H0
		Hctrl = self.qh_in.Hctrl

		u_all = np.zeros((tim_all+1,dim,dim),dtype = complex)
		u_all[-1,:,:] = np.eye(dim)

		for tim in xrange(tim_all,0,-1):
			H = H0 + np.matrix( ctrl[tim-1] * np.array(Hctrl) )
			u_all[tim-1,:,:] = self.u_prev(H, u_all[tim,:,:])

		return u_all

	def psi_t(self):
		"""backward state start from time T with goal state"""
		dim = self.qh_in.dim
		tim_all = self.qh_in.tim_all 
		psi_all = np.zeros((tim_all+1,1,dim), dtype = complex)
		psi_all[-1,:,:] = np.matrix(self.phi_go[:]).T
		u_all = self.u_t_back()

		for tim in xrange(tim_all,0,-1):
			psi_all[tim,:,:] = np.dot(psi_all[-1,:,:], u_all[tim,:,:])
		return psi_all

	def d_ctrl(self, psi_now, Hctrl, phi_now):
		"""calculate new control/laser variation"""
		return np.real(np.dot(psi_now, np.dot(Hctrl, phi_now)))

	def norm_ctrl(self, *ctrls):
		"""normalize to unit one of control"""
		ctrl_norm = 0
		for ctrl in ctrls:
			ctrl_norm += ctrl**2

		return ctrls/ctrl_norm	
	
	def fidelity(self, phi_T, phi_go):
		"""fidelity of phi at final time T """
		return np.dot(np.matrix(phi_go).T,phi_T)/np.dot(np.matrix(phi_go).T,phi_go)

	def run(self):
		"""run quantum optimal control algoritm"""
		start = clock()
		ctrl = self.qh_in.ctrl
		phi_t  = self.qh_in.phi_t() 
		tim_all = self.qh_in.tim_all
		iter_time = self.iter_time
		phi_go = self.phi_go
		H0 = self.qh_in.H0
		Hctrl = self.qh_in.Hctrl
		lmda = self.lmda

		for it in range(iter_time):

			psi_t = self.psi_t()
			fi = self.fidelity(phi_t[-1,:,:], phi_go[:])
			print 'IterTime: %s,   Error: %s,   TotTime: %s,   AvgTime: %s'\
				 %( it+1, 1-abs(fi), clock()-start, (clock()-start)/(it+1))
			
			if 1-abs(fi) < self.error_bd:
				break
		
			for tim in range(tim_all):
				dctrl = self.d_ctrl(psi_t[tim,:,:], Hctrl, phi_t[tim,:,:])\
						/(2*lmda)
				ctrl[tim] += dctrl 
				H = H0 + np.matrix( ctrl[tim] * np.array(Hctrl) )
				u_next  = self.qh_in.u_dt(H)
				phi_t[tim+1,:,:] = np.dot(u_next, phi_t[tim,:,:])


		return ctrl 	
		

if __name__ == '__main__':

	H0 = [[1,1],[1,1]]
	Hctr = [[1,0],[0,-1]]
	ctrl_i = .1*np.ones(1000)
	phi_i = [[0],[1]]
		
	qh_test = QH(H0, Hctr, ctrl_i, phi_i)
	time = qh_test.real_tim
		
	phi = qh_test.phi_t()
	prob = qh_test.prob_t(phi)
	plt.plot(time, prob[:,0,:],'r')
	plt.plot(time, prob[:,1,:],'b')
	plt.show()
	
	
	phi_g = [[1],[0]]
	qoct_test = QOCT(qh_test, phi_g)
	
	"""
	psi = qoct_test.psi_t()
	prob_s = psi*np.conjugate(psi)

	plt.plot(time, prob_s[:,0,:],'r')
	plt.plot(time, prob_s[:,1,:],'b')
	plt.show()
	"""
	
	ctrl_test = qoct_test.run()	
	plt.plot(time[:-1], ctrl_test)
	plt.show()
	
	phi_new = qh_test.phi_t()
	prob_new = qh_test.prob_t(phi_new)
	
	plt.plot(time, prob_new[:,0,:],'r')
	plt.plot(time, prob_new[:,1,:],'b')
	plt.show()
	
	lon = np.size(ctrl_test)
	ctrl_lon = np.zeros(3*lon)
	ctrl_lon[lon:2*lon ] = ctrl_test[:]

	qh_test2 = QH(H0, Hctr, ctrl_lon, phi_i)
	time2 = qh_test2.real_tim
	phi2 = qh_test2.phi_t()

	prob2 = qh_test2.prob_t(phi2)
	plt.plot(time2, prob2[:,0,:],'r')
	plt.plot(time2, prob2[:,1,:],'b')
	plt.show()
	

