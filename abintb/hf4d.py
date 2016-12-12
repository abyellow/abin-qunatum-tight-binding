import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
from time import time


class Hf:

	def __init__(self, v0=0., vf=1., E0=1., freq = 3., m_max=10, knum = 100):

		self.v0 = 0
		self.vf = vf
		self.E0 = E0
		self.freq = freq
		self.m_max = m_max
		self.knum = knum
		self.eps = np.array(range(self.knum+1))*-2*np.pi/(self.knum) + np.pi 
		self.phase = 0

	def gcoeff(self,k,m,n,E0):

		tau = self.tau
		deltau = self.deltau
		Ra = tau+deltau
		rb = tau-deltau
		if m==n:
			delta = 1
		else:
			delta = 0

		phase = np.exp(1j*(m-n)*np.pi*-1./2)

		epsk = np.exp(1j*k)
		bessel = ss.jv(m-n,E0) 
		gk_mn  = (Ra*delta + rb*epsk*bessel ) * phase
		return gk_mn

	
	def Hfloq(self,k):
		
		E0 = self.E0
		freq = self.freq
		m_max = self.m_max
		v = np.array(range(-m_max,m_max+1))*freq
		v2 = np.repeat(v,2)
		Hf = np.diag(v2) * (1+0j)
		for i in range(2*m_max+1):
			for j in range(2*m_max+1):
				idxi= 2*i
				idxj = 2*j+1
				mi = i - m_max 
				mj = j - m_max
				Hf[idxi,idxj] = self.gcoeff(k,mi,mj,E0)
				Hf[idxi+1,idxj-1] = np.conj(Hf[idxi,idxj])
		return Hf

	def eigV(self,k):
		
		w, v = np.linalg.eigh(self.Hfloq(k))
		return v

	def berry(self):
		m_max = self.m_max
		eps = self.eps
		vec = np.array(map(self.eigV,eps))
		berry = np.zeros(np.shape(vec)[1])
		berry_size = len(berry)
		for j in range(berry_size):
			m_j = j/2.- m_max
			berry[j] += np.sum(self.berry_band(m_j))
		return berry

	def berry_band(self,m):
		
		vec = np.array(map(self.eigV,self.eps))
		vec_size = np.shape(vec)[1]
		bandm = int(vec_size/2. + 2*m - 1)
		phk = np.exp(1j*np.angle(vec[:,0,bandm]))

		for j in range(len(vec[0,:,bandm])):
			vec[:,j,bandm] = vec[:,j,bandm]/phk
		
		berry = []
		for i in range(len(self.eps)-2):
			berry.append(np.dot(np.conj(vec[i+1,:,bandm]),vec[i+2,:,bandm]-vec[i,:,bandm]))
		berry = np.array(berry)
		return np.real(1j*berry)/2.

	def eigE(self,k):
		
		w, v = np.linalg.eigh(self.Hfloq(k))
		return w

	def spec(self):

		eps = self.eps
		spec = np.array(map(self.eigE,eps))
		return spec

	def spec_plot(self,spec,style = 'k-', mark=True):

		eps = self.eps
		for i in range(np.shape(spec)[1]):
			plt.plot(eps,spec[:,i],style,linewidth=2.5)
		
		if mark:
			plt.ylim([-3.142,3.142])
			plt.xlim([-3.142,3.142])
			xlab = r' $k $'
			ylab = r'Energy / $\tau$'
			fsize = 16
			plt.xlabel(xlab, fontsize = fsize)
			plt.ylabel(ylab, fontsize = fsize)
			plt.title( 'deltau = %.2f, freq = %.2f, E0 = %.2f, phase = %d'
					%(self.deltau, self.freq, self.E0,self.phase))

		return 0




if __name__=='__main__':

	tau = 1.
	deltau = .5
	m_max = 0 
	E0 = 1.
	freq = 3.
	phase = 2
	k_num = 100 
	option = 0

	import argparse
	pa = argparse.ArgumentParser()
	pa.add_argument('--f', type = float)
	pa.add_argument('--E', type = float)
	pa.add_argument('--dt', type = float)
	pa.add_argument('--ph', type = int)
	pa.add_argument('--m', type = int)
	pa.add_argument('--k', type = int)
	pa.add_argument('--opt', type = int)
	args = pa.parse_args()

	if args.f:
		freq = args.f
	if args.E:
		E0 = args.E
	if args.dt:
		deltau = args.dt
	if args.ph:
		phase = args.ph
	if args.m:
		m_max = args.m
	if args.k:
		k_num = args.k
	if args.opt:
		option = args.opt

	cond = 'Conditions: deltau = %.2f, freq = %.2f, E0 = %.2f, m = %d, knum = %d'\
		 %(deltau, freq, E0, m_max, k_num)
	print cond

	hf = SSHHf(deltau = deltau, m_max = m_max, freq = freq, E0 = E0, phase = phase, knum=k_num)
	hf2 = SSHHf(deltau = -deltau, m_max = m_max, freq = freq, E0 = E0, phase = phase, knum=k_num)
		
	st = time()	

	#spectrum
	if option == 0:
		hf.spec_plot(hf.spec())
		hf2.spec_plot(hf2.spec(),style = 'k--')
		plt.show()

	#same condtions, berry phase for all bands
	elif option == 1:	
		x =  hf.eigE(np.pi)
		berryPh1 = hf.berry()
		berryPh2 = hf2.berry()
		print np.round(berryPh1/(-np.pi),1)
		print np.round(berryPh2/(-np.pi),1)
		by1 = np.round(berryPh1/-np.pi) % 2
		by2 = np.round(berryPh2/-np.pi) % 2

		plt.ylim(-1.3,1.3)
		plt.plot(x,by1,'^-',label=r'$\delta\tau>0$')
		plt.plot(x,-by2,'v-',label=r'$\delta\tau<0$')
		plt.xlabel(r'Band energy',fontsize=16)
		plt.ylabel(r'$\gamma/\pi$',fontsize=16)
		plt.title(cond)
		plt.legend()
		plt.show()

	#single band berry connection value and berry phase transition.
	elif option == 2 :
		
		x =  hf.eps[:-2]
		by1 = hf.berry_band(m=0)
		by2 = hf2.berry_band(m=0)

		plt.plot(x,by1,'^-',label=r'$\delta\tau>0$')
		plt.plot(x,-by2,'v-',label=r'$\delta\tau<0$')
		plt.legend()
		plt.show()

		by1_int = np.zeros(len(by1)+1)
		by2_int = np.zeros(len(by2)+1)
		for i in range(len(by1)):
			by1_int[i+1] = by1_int[i] + by1[i]
			by2_int[i+1] = by2_int[i] + by2[i]

		plt.plot(by1_int/-np.pi,'^-',label=r'$\delta\tau>0$')
		plt.plot(by2_int/-np.pi,'v-',label=r'$\delta\tau<0$')
		plt.grid(b=True, which='major', color='b', linestyle='--')
		plt.legend()
		plt.show()

		print np.round(np.sum(by1)/-np.pi,2), np.round(np.sum(by2)/-np.pi,2)

	#energy band vs amplitude E0
	elif option == 3:

		freq_list = [1.6,3,1.6,3]
		deltau_list = [.5,.5,-.5,-.5]	
		style_list = ['ro-','g^-','bs--','cd--']
		E_list = np.linspace(0,5,51)

		for i in range(len(freq_list)):
			freq = freq_list[i]
			deltau = deltau_list[i]
			style = style_list[i]
			e1_up = []
			e1_dn = []
			bandm = 2*m_max
			bandn = bandm+1

			for E0 in E_list:

				hf1 = SSHHf(deltau = deltau, m_max = m_max, freq = freq, E0 = E0, phase = phase, knum=k_num)
				e1 = hf1.eigE(k=0.001)

				e1_up.append(e1[bandm])
				e1_dn.append(e1[bandn])

			plt.plot(E_list,e1_up,style,label=r'$\delta\tau = %.1f$, $\Omega = %.1f$'%(deltau,freq))
			plt.plot(E_list,e1_dn,style)#,label=r'$\delta\tau>0$')

		plt.xlabel(r'$A_0$',fontsize = 16)
		plt.ylabel(r'Energy / $\tau$',fontsize = 16)
		plt.gca().xaxis.grid(True)
		#plt.title(cond)
		plt.legend()
		plt.show()
	print 'run_time: ', time()-st
