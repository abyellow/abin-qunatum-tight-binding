#import sys
#sys.path.append('/home/abin/abin/projects/abin-tight-binding/')

import numpy as np
from time import time
import matplotlib.pyplot as plt
#from matplotlib.colors import SymLogNorm
#from sshHf import SSHHf
import qn, tb


class qn4d(qn.QnIni):

	def __init__(self, k, ctrlt, dt=.1, v0 = 1., vf = 1.):

		self.ctrlt = ctrlt
		self.k = k 
		self.k0 = [0,0,0,0]
		self.dt = dt
		self.v0 = v0
		self.vf = vf
		self.H0 = np.zeros((4,4))
		#qn.QnIni.__init__(self,k=0,ctrlt=np.zeros(100))
		self.save_name = 'cone_save_name'

	def dvec(self,ctrl):

		k0 = self.k0
		k  = self.k
		v0 = self.v0
		vf = self.vf

		di = v0 * (k[0] - ctrl - k0[0])
		dx = vf * (k[1] - ctrl - k0[1])
		dy = vf * (k[2] - ctrl - k0[2])
		dz = vf * (k[3] - ctrl - k0[3])

		return [di,dx,dy,dz]

	def ham(self,ctrl):

                pau_i = np.array([[1,0],[0,1]])
                pau_x = np.array([[0,1],[1,0]])
                pau_y = np.array([[0,-1j],[1j,0]])
                pau_z = np.array([[1,0],[0,-1]])

                d = self.dvec(ctrl)
                cone = pau_i * d[0] + pau_x * d[1] + pau_y * d[2] + pau_z * d[3]
		m0 = 0. * np.identity(2)
		h = np.bmat([[cone,m0],[m0,-cone]])#.reshape((4,4))

		return h


if __name__=='__main__':

	dt = .1
	tnum = 400
	t_rel = np.linspace(-20,20,tnum-1)
	ctrlt = np.cos(t_rel)
	cond1 = qn4d(k=[0,1,2,3],ctrlt = ctrlt)
	'''
	print a.ham(ctrl=0)
	print np.shape(a.ham_t())
	print 'pass 1'
	print np.shape(a.phi_i())
	print a.eig_energy()
	print a.save_name
	'''
	model1 = qn.QnModel(cond1)
	phit = model1.phi_t()
	probt = model1.prob_t(phit)
	'''
	print np.shape(probt)
	for i in range(4):
		plt.plot(t_rel,probt[:-1,i,:])
	plt.show()
	'''
	ti = time()
	knum = 100
	keps = np.linspace(-np.pi,np.pi,knum)
	k0 = np.zeros(knum) +0.001
	kall = np.array([k0,k0,k0,keps])#.T
	#print np.shape(kall)
	tb1 = tb.TbModel(cond1,kall,model_dim = 4)
	tb1.tb_model= False
	ckall = np.array(tb1.phi_kall())
	print ckall.shape
	print 'run_time: ', time()-ti
