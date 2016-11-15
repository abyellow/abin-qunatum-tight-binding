"""
 Editor Bin H.
 Quantum Optimal Control Example of Two Control Parameters and Normalization

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from time import clock

class QH:
	"""
	Initial data/conditions of Quantum Hamiltonian and initial states.
	"""
	def __init__(self, H0, Hctrl, ctrl_i, Hctrl2, ctrl_i2, phi_i, dt=.01):

		self.H0 = H0          #Hamiltonian with no control/laser
		self.Hctrl = Hctrl    #Hamiltonian of control/laser term
		self.ctrl = ctrl_i    #initial control/laser
		self.Hctrl2 = Hctrl2    #Hamiltonian of control/laser term
		self.ctrl2 = ctrl_i2    #initial control/laser
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
		ctrl2 = self.ctrl2
		H0 = self.H0
		Hctrl = self.Hctrl
		Hctrl2 = self.Hctrl2

		u_all = np.zeros((tim_all+1,dim,dim),dtype = complex)
		u_all[0,:,:] = np.eye(dim)

		for tim in xrange(tim_all):
			H = H0 + np.matrix( ctrl[tim] * np.array(Hctrl)\
						 + ctrl2[tim] * np.array(Hctrl2) )
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

	def prob_t(self, phi):
		"""probability distribution of state phi"""
		return np.real(phi*np.conjugate(phi))

	def eigE_t(self):
		"""Caculate eigen energy variation with time"""
		tim_all = self.tim_all
		ctrl = self.ctrl
		ctrl2 = self.ctrl2
		H0 = self.H0
		Hctrl = self.Hctrl
		Hctrl2 = self.Hctrl2
		eig_E = []
		for tim in xrange(tim_all):
			H = H0 + np.matrix( ctrl[tim] * np.array(Hctrl)\
						 + ctrl2[tim] * np.array(Hctrl2) )
			eig_val, eig_vec = np.linalg.eig(H)
			eig_E.append(eig_val)

		return np.array(eig_E)



class QOCT:
	"""
	Quantum optimal control codes
	"""

	def __init__(self, qh_input, phi_g, lmda = 10.):
		
		self.error_bd = 10**-4  # error bound of convergence 
		self.qh_in = qh_input   # class QH for all i.c. and EoM
		self.phi_g = phi_g      # goal quantum states we expect
		self.lmda = lmda         # learning rate
		self.iter_time = 1000


	
	def u_prev(self, H, u_now):
		"""Derive U at next time step"""
		return np.dot(u_now, self.qh_in.u_dt(H))

	def u_t_back(self):
		"""Evolve propergator backward for given time period"""
		dim = self.qh_in.dim 
		tim_all = self.qh_in.tim_all
		ctrl = self.qh_in.ctrl
		ctrl2 = self.qh_in.ctrl2
		H0 = self.qh_in.H0
		Hctrl = self.qh_in.Hctrl
		Hctrl2 = self.qh_in.Hctrl2

		u_all = np.zeros((tim_all+1,dim,dim),dtype = complex)
		u_all[-1,:,:] = np.eye(dim)

		for tim in xrange(tim_all,0,-1):
			H = H0 + np.matrix( ctrl[tim-1] * np.array(Hctrl)\
						 + ctrl2[tim-1] * np.array(Hctrl2) )
			u_all[tim-1,:,:] = self.u_prev(H, u_all[tim,:,:])

		return u_all

	def psi_t(self):
		"""backward state start from time T with goal state"""
		dim = self.qh_in.dim
		tim_all = self.qh_in.tim_all 
		psi_all = np.zeros((tim_all+1,1,dim),dtype = complex)
		psi_all[-1,:,:] = np.matrix(self.phi_g[:]).T
		u_all = self.u_t_back()

		for tim in xrange(tim_all,0,-1):
			psi_all[tim,:,:] = np.dot(psi_all[-1,:,:], u_all[tim,:,:])
		return psi_all

	def d_ctrl(self, psi_now, Hctrl, phi_now):
		"""calculate new control/laser variation"""
		return np.real(np.dot(psi_now, np.dot(Hctrl, phi_now)))

	def ctrl_norm(self, ctrl, ctrl2):
		"""normalize to unit one of controls"""
		return  np.sqrt(ctrl**2 + ctrl2**2)
	
	def fidelity(self, phi_T, phi_g):
		"""fidelity of phi at final time T """
		return np.dot(np.matrix(phi_g).T,phi_T)/np.dot(np.matrix(phi_g).T,phi_g)

	def run(self):
		"""run quantum optimal control algoritm"""
		start = clock()
		ctrl = self.qh_in.ctrl
		ctrl2 = self.qh_in.ctrl2
		phi_t  = self.qh_in.phi_t() 
		tim_all = self.qh_in.tim_all
		iter_time = self.iter_time		
		phi_g = self.phi_g
		H0 = self.qh_in.H0
		Hctrl = self.qh_in.Hctrl
		Hctrl2 = self.qh_in.Hctrl2
		lmda = self.lmda


		for it in xrange(iter_time):
			
			psi_t = self.psi_t()
			fi = (self.fidelity(phi_t[-1,:,:], phi_g[:]))
			print 'IterTime: %s,   Error: %s,   TotTime: %s,   AvgTime: %s'\
				%( it+1, 1-abs(fi), clock()-start, (clock()-start)/(it+1))
			
			if 1-abs(fi) < self.error_bd:
				break
		
			for tim in xrange(tim_all):

				dctrl = self.d_ctrl(psi_t[tim,:,:], Hctrl, phi_t[tim,:,:])\
						/(2*lmda)
				dctrl2 = self.d_ctrl(psi_t[tim,:,:], Hctrl2, phi_t[tim,:,:])\
						/(2*lmda)
				ctrl[tim] += dctrl 
				ctrl2[tim] += dctrl2
				ctrl_norm = self.ctrl_norm(ctrl[tim], ctrl2[tim])
				ctrl[tim], ctrl2[tim] = ctrl[tim] / ctrl_norm, ctrl2[tim] / ctrl_norm

				H = H0 + np.matrix( ctrl[tim] * np.array(Hctrl) \
							+ ctrl2[tim] * np.array(Hctrl2) )
				u_next  = self.qh_in.u_dt(H)
				phi_t[tim+1,:,:] = np.dot(u_next, phi_t[tim,:,:])


		return ctrl, ctrl2 	
		

if __name__ == '__main__':

	H0 = np.matrix([[1,1],[1,-1]])
	Hctr = [[1,0],[0,-1]]
	Hctr2 = [[1,0],[0,-1]]
	ctrl = .9*np.ones(1000)
	ctrl2 = .1*np.ones(1000)
	
	norm = lambda x: np.sqrt(sum(np.array(x)**2))
	 
	phi_i = [[1],[np.sqrt(2)-1]]
	phi_i = phi_i / norm(phi_i)
	
	
	qh_test = QH(H0, Hctr, ctrl, Hctr2, ctrl2, phi_i)
	time = qh_test.real_tim
		
	phi = qh_test.phi_t()
	prob = qh_test.prob_t(phi)
	plt.plot(time, prob[:,0,:],'r')
	plt.plot(time, prob[:,1,:],'b')
	plt.show()
	
	eigE = qh_test.eigE_t()
	plt.plot(time[:-1], eigE[:,0],'r')
	plt.plot(time[:-1], eigE[:,1],'b')
	plt.show()
	
	
	phi_g = [[np.sqrt(2)-1],[-1]]
	phi_g = phi_g / norm(phi_g) 
	qoct_test = QOCT(qh_test,phi_g)


	ctrl_test, ctrl2_test = qoct_test.run()	
	plt.plot(time[:-1], ctrl_test)
	plt.plot(time[:-1], ctrl2_test)
	plt.show()
	
	phi_new = qh_test.phi_t()
	prob_new = qh_test.prob_t(phi_new)#phi_new*np.conjugate(phi_new)
	
	plt.plot(time, prob_new[:,0,:],'r')
	plt.plot(time, prob_new[:,1,:],'b')
	plt.show()
	
	lon = np.size(ctrl_test)
	ctrl_lon = np.zeros(3*lon)
	ctrl_lon[lon:2*lon] = ctrl_test[:]

	ctrl2_lon = np.zeros(3*lon)
	ctrl2_lon[lon:2*lon] = ctrl2_test[:]

	qh_test2 = QH(H0, Hctr, ctrl_lon, Hctr2, ctrl2_lon, phi_i)

	time2 = qh_test2.real_tim
		
	phi2 = qh_test2.phi_t()
	prob2 = qh_test2.prob_t(phi2)	
	plt.plot(time2, prob2[:,0,:],'r')
	plt.plot(time2, prob2[:,1,:],'b')
	plt.show()
	

