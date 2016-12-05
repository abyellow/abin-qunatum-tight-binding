#import sys
#sys.path.append('/home/abin/abin/projects/abin-tight-binding/')

import numpy as np
from time import time
import matplotlib.pyplot as plt
#from matplotlib.colors import SymLogNorm
#from sshHf import SSHHf
import qn, tb


class qn4d:#(qn.QnIni):

	def __init__(self, k, ctrlt, dt=.1, v0 = 1., vf = 1.,state='mix'):

		self.ctrlt = ctrlt
		self.k = k 
		self.k0 = [0,0,0,0]
		self.dt = dt
		self.v0 = v0
		self.vf = vf
		self.H0 = np.zeros((4,4))
		#qn.QnIni.__init__(self,k=0,ctrlt=np.zeros(100))
		self.state = state
		self.save_name = 'cone_save_name'

	def dvec(self,ctrlx,ctrly):

		k0 = self.k0
		k  = self.k
		v0 = self.v0
		vf = self.vf

		di = v0 * (k[0] - k0[0])
		dx = vf * (k[1] - ctrlx - k0[1])
		dy = vf * (k[2] - ctrly - k0[2])
		dz = vf * (k[3] - k0[3])

		return [di,dx,dy,dz]

	def ham(self,ctrlx,ctrly):

                pau_i = np.array([[1,0],[0,1]])
                pau_x = np.array([[0,1],[1,0]])
                pau_y = np.array([[0,-1j],[1j,0]])
                pau_z = np.array([[1,0],[0,-1]])

                d = self.dvec(ctrlx,ctrly)
                cone = pau_i * d[0] + pau_x * d[1] + pau_y * d[2] + pau_z * d[3]
		m0 = 0. * np.identity(2)
		h = np.bmat([[cone,m0],[m0,-cone]])#.reshape((4,4))

		return h

	def ham_t(self):
		ctrlt = self.ctrlt
		return np.array(map(self.ham,ctrlt[0],ctrlt[1]))


	def phi_i(self):

		state= self.state
		w,v = np.linalg.eigh(self.ham(ctrlx=0,ctrly=0))

		if state == 'down':
			return v[:,0].reshape(len(v[:,0]),1)

		elif state == 'down2':
			return v[:,1].reshape(len(v[:,1]),1)

		elif state == 'up':
			return v[:,2].reshape(len(v[:,1]),1)

		elif state == 'up2':
			return v[:,3].reshape(len(v[:,1]),1)

		elif state == 'mix':
			return ((v[:,0]+v[:,1])/np.sqrt(2)).reshape(len(v[:,0]),1)

		elif state == 'mix2':
                        return ((v[:,0]+v[:,1]+v[:,2]+v[:,3])/np.sqrt(4)).reshape(len(v[:,0]),1)
		else: 
			print 'no such state!!'


	def eig_energy(self,ctrlx=0,ctrly=0):
		w, v = np.linalg.eigh(self.ham(ctrlx,ctrly))
		return w




if __name__=='__main__':

	dt = .1
	tnum = 400
	t_rel = np.linspace(-20,20,tnum-1)
	ctrlt = [np.cos(t_rel),np.sin(t_rel)]
	
	cond1 = qn4d(k=[0,1,2,3],ctrlt = ctrlt,state='mix')
	print cond1.phi_i()
	cond1.state = 'down'
	print cond1.phi_i()
	cond1.state = 'down2'
	print cond1.phi_i()
	cond1.state = 'up'
	print cond1.phi_i()
	cond1.state = 'up2'
	print cond1.phi_i()
	cond1.state = 'mix2'
	#print cond1.phi_i(state='mix2')
	#print cond1.phi_i(state='mix4')



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
	knum = 10
	keps = np.linspace(-np.pi,np.pi,knum)
	k0 = np.zeros(knum) +0.001
	kall = np.array([k0,k0,k0,keps])#.T
	#print np.shape(kall)
	tb1 = tb.TbModel(cond1,kall,model_dim = 4)
	tb1.tb_model= False
	ckall = np.array(tb1.phi_kall())
	print ckall.shape
	print 'run_time: ', time()-ti
