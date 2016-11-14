import numpy as np
from time import time
#from itertools import product
import sys
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
from sshPES import PES
from sshIniData import SSHIniData
from sshHf import SSHHf 
#from mpl_toolkits.mplot3d import Axes3D

class pseudoSpin(PES):
	
	def integral_PES(self,PESx,PESy,lb,ub,rang,form,line = True):

		w_num = self.w_num
		PESy = np.array(PESy)
		PESx = np.array(PESx)

		pl = int((np.pi-lb)/(2*np.pi) * w_num)
		pu = int((np.pi-ub)/(2*np.pi) * w_num)
		pr = int((rang)/(2*np.pi) * w_num)+1
		absPESy = abs(PESy)
		intPESy = []
		intPESx = []
		if form == 'box':
			for i in range(len(PESy[0,:])):
				intPESy.append(np.sum(PESy[pu:pl,i],axis=0))
				intPESx.append(np.sum(PESx[pu:pl,i],axis=0))
				if line:	
					PESx[pu,i] = 100
					PESy[pl,i] = 100
					PESx[pl,i] = 100
					PESy[pu,i] = 100

		elif form == 'wave':
			for i in range(len(PESy[0,:])):
				ind = pu + np.argmax(absPESy[pu:pl,i])
				intPESy.append(np.sum(PESy[ind-pr:ind+pr,i],axis=0))
				intPESx.append(np.sum(PESx[ind-pr:ind+pr,i],axis=0))
				if line:
					PESx[ind-pr,i] = 100
					PESy[ind-pr,i] = 100
					PESx[ind+pr,i] = 100
					PESy[ind+pr,i] = 100

		intPESx.append(intPESx[0])
		intPESy.append(intPESy[0])
		return np.array(intPESx), np.array(intPESy), PESx, PESy

	def phase(self,x,y):
	
		phi = np.sign(y)*np.arccos(x/np.sqrt(x**2+y**2))
		phb = phi[-1]-phi[0]

		x,y = self.norm_spin(x,y,factor=False)

		ox = y[1:]-y[:-1]
		oy = -(x[1:]-x[:-1])
		drx = (x[1:]+x[:-1])/2.
		dry = (y[1:]+y[:-1])/2.
		pha = np.sum(ox*drx + oy*dry)
		
		return pha, phb

	def norm_spin(self,x,y,factor=True):
		fac = 1
		if factor:
			fac = np.linspace(1,2,len(x))

		r = np.sqrt(x**2+y**2)
		x, y = fac*x/r, fac*y/r
		#x=np.insert(x,0,1)
		#y=np.insert(y,0,0)
		#x=np.insert(x,len(x),2)
		#y=np.insert(y,len(y),0)
		return x, y

	def plot_spin(self,x,y, norm = True):

		if norm:
			x, y = self.norm_spin(x,y)

		plt.figure()
		plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
		#plt.plot(x, y, 'o-',linewidth=1.5)
		plt.plot(0,0,'x',markersize = 20)
		plt.plot(np.linspace(-3,3,len(x)),np.zeros(len(x)),'g--')
		plt.xlabel('Px')
		plt.ylabel('Py')
		plt.xlim([-2.1,2.1])
		plt.ylim([-2.1,2.1])
		#plt.title('tp = %.1f, E0 = %.1f, Freq = %.2f, dt = %.1f' %(tp0,E0,freq,deltau))
		#plt.savefig('figure/sshspin_spin.png')

	def phase_PES(self,PESx,PESy):

		PESint = np.sqrt(PESx**2 + PESy**2)
		PESphi = np.pi*(1-np.sign(PESy))/2 + np.arccos(PESx/PESint) - np.pi
		print np.amax(PESphi)
		return PESint*PESphi/(4*np.pi)


if __name__ == "__main__":

	tp0 = 0.
	E0 = 1. 
	freq = 3. 
	tau = 1. 
	deltau = .5
	paui = 'x'
	pauj = 'y'
	k_num = 12*6 

	width = 50
	h_choose = 0
	band = 'mix'
	option = 0
	int_form = 'box'
	int_lb = -1.6#0#-3.#2.0
	int_ub = -.8#.8#0#-2.#3.0
	int_rang = .25
	int_line = False

	import argparse
	pa = argparse.ArgumentParser()
	pa.add_argument('--f', type = float)
	pa.add_argument('--E', type = float)
	pa.add_argument('--dt', type = float)
	pa.add_argument('--tp', type = int)
	pa.add_argument('--model', type = int)
	pa.add_argument('--k', type = int)
	pa.add_argument('--opt', type = int)
	pa.add_argument('--wid', type = int)
	pa.add_argument('--lb', type = float)
	pa.add_argument('--ub', type = float)
	args = pa.parse_args()

	if args.f:
		freq = args.f
	if args.E:
		E0 = args.E
	if args.dt:
		deltau = args.dt
	if args.tp:
		tp0 = args.tp
	if args.model:
		h_choose = args.model
	if args.k:
		k_num = args.k
	if args.opt:
		option = args.opt
	if args.wid:
		width = args.wid
	if args.lb:
		int_lb = args.lb
	if args.ub:
		int_ub = args.ub
	cond = 'Conditions: model = %d, deltau = %.2f, freq = %.2f, E0 = %.2f, tp0 = %d, knum = %d, p_width = %d'\
		 %(h_choose, deltau, freq, E0, tp0, k_num,width)
	print cond

	t_s = -200
	dt = .1
	std1 = 15 
	t_in = t_s-std1*3

	#print 'width of pulse: ',width #/ np.sqrt(2)
	n_tot = int(-2*t_s/dt) + int(std1*6/dt)
	t_rel = (np.array(range(n_tot-1)))*dt + t_in
	ctrl = np.exp(-.5*(t_rel/width)**2) * E0 * np.sin(freq*t_rel)

	m_max = 10
	hf = SSHHf(deltau = (-1)**(h_choose) * deltau, m_max = m_max, freq = freq, E0 = E0, phase = 2, knum=k_num)
	def plot_hf():
		eps = hf.eps
		spec = hf.spec()
		nk = k_num
		for i in range(np.shape(spec)[1]):
			plt.plot(eps/2.,spec[:,i],'k--',linewidth=2.)
		plt.xlim([-3.15/2.,3.15/2.])
		plt.ylim([-3.15,3.15])



	timea  = time()
	init = SSHIniData(tau, deltau, ctrl, knum=k_num, dt=dt, ham_choose = h_choose, iniband= band)
	cvec1 = init.clc_cvec()
	PES_spin = pseudoSpin(init, cvec1, tin = t_in, E0=E0, freq=freq, std = std1, width = 1) 


	try:
		load_i = 'data/PES2ssh_ham_%d_dt_%.2f_ktimes_%d_tp_%.1f_E0_%.1f_freq_%.2f_deltau_%.1f_paui_%s_std_%.1f_band_%s.txt'\
				%(h_choose, dt,k_num,tp0,E0,freq,deltau,paui,std1,band)
		PESloadi = np.loadtxt(load_i)[::-1]
		print "file1 exist, loading====>"

	except IOError:
		print load_i
		print "no such file, calculating====>"
		PESloadi = PES_spin.final_run(tp = tp0, pau = paui)[::-1]


	try:
		load_j = 'data/PES2ssh_ham_%d_dt_%.2f_ktimes_%d_tp_%.1f_E0_%.1f_freq_%.2f_deltau_%.1f_paui_%s_std_%.1f_band_%s.txt'\
				%(h_choose, dt,k_num,tp0,E0,freq,deltau,pauj,std1,band)
		PESloadj = np.loadtxt(load_j)[::-1]
		print "file2 exist, loading====>"
	except IOError:
		print load_j
		print "no such file, calculating====>"
		PESloadj = PES_spin.final_run(tp = tp0, pau = pauj)[::-1]
#

	timea = time()
	PES2Dx = PESloadi
	PES2Dy = PESloadj

	print 'total time:', time()-timea

	if option == 1:
		x, y, PES2Dx, PES2Dy = PES_spin.integral_PES(PES2Dx,PES2Dy,int_lb,int_ub,int_rang,form=int_form,line = True)
		PES_spin.plot_spin(x,y)
		plt.title(cond)
		plt.show()

		x1,y1 = PES_spin.norm_spin(x,y,factor=False)
		savename = 'data/spin_ham_%d_dt_%.2f_ktimes_%d_tp_%.1f_E0_%.1f_freq_%.2f_deltau_%.1f_std_%.1f_band_%s_lb_%.1f_ub_%.1f.txt'\
				%(h_choose, dt,k_num,tp0,E0,freq,deltau,std1,band,int_lb,int_ub)
		np.savetxt(savename,zip(x1,y1))

		pha, phb = PES_spin.phase(x,y)
		ra = np.round(pha/(2*np.pi),1) %2
		rb = np.round(phb/(2*np.pi),1) %2
		print 'phase number: ',pha, phb, ra, rb


	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	PES_spin.plot(PES2Dx,ax1)
	plot_hf()
	ax2 = fig.add_subplot(122)
	PES_spin.plot(PES2Dy,ax2)
	plot_hf()
	fig.suptitle(cond)
	plt.tight_layout()
	plt.savefig('figure/sshspin_pxpy.png')
	plt.show()


	if option == 2:
		PESphase = PES_spin.phase_PES(PES2Dx,PES2Dy)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		PES_spin.plot(PESphase,ax,color='hsv')
		plot_hf()
		plt.tight_layout()
		plt.savefig('figure/sshspin_phase.png')
		plt.show()

