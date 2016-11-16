import numpy as np
import matplotlib.pyplot as plt
from time import time

from qn import QnIni, QnModel


class TbModel:

	def __init__(self, QnIni, kall, model_dim = 1):

		self.QnIni = QnIni
		self.kall = np.array(kall)
		self.model_dim = model_dim 
		#self.save_name  = 'name'		

	def phi_kall(self):

		kall = self.kall
		model_dim = self.model_dim
		if model_dim ==1:
			cvec = np.array(map(self.phi_k1D, kall))
		elif model_dim ==2:
			cvec = np.array(map(self.phi_k2D, kall[0], kall[1]))

		return np.swapaxes(cvec[:,:,:,0],1,2)


	def phi_k1D(self,kx):
		
		self.QnIni.k = kx 
		qnmodel = QnModel(self.QnIni, tb_model = True)
		phit = qnmodel.phi_t()

		return phit


	def phi_k2D(self,kx,ky):

		self.QnIni.k = [kx,ky]
		self.QnIni.ham_val = 4
		qnmodel = QnModel(self.QnIni, tb_model = True)
		phit = qnmodel.phi_t()

		return phit


	def spec(self):

		kall = self.kall
		spec = np.array(map(self.QnIni.eig_energy,kall))
		return spec


if __name__ == "__main__":

	dt = .1
	E0 = 1. 
	knum = 100 
	freq = 1. 

	tau = 1. 
	deltau = .5#-.3
	keps = np.linspace(-np.pi,np.pi,knum)

	n_tot = 4000 
	t_rel = (np.array(range(n_tot-1))-2000)*dt
	ctrli =  E0 * np.cos(freq*t_rel)
	ki = 0.0001
	cond1 = QnIni(k=ki, ctrlt=ctrli)

	ti = time()
	tb1 = TbModel(cond1, keps)	
	ckall = np.array(tb1.phi_kall())
	print ckall.shape
	print 'run_time: ', time() - ti

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

