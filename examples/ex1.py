import sys
sys.path.append('/home/abin/abin/research/abin-tight-binding/')

from time import time
import numpy as np
import matplotlib.pyplot as plt

import abintb.hf as hf 

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

	hf1 = hf.Hf(deltau = deltau, m_max = m_max, freq = freq, E0 = E0, phase = phase, knum=k_num)
	hf2 = hf.Hf(deltau = -deltau, m_max = m_max, freq = freq, E0 = E0, phase = phase, knum=k_num)
		
	st = time()	

	#spectrum
	if option == 0:
		hf1.spec_plot(hf1.spec())
		hf2.spec_plot(hf2.spec(),style = 'k--')
		plt.show()
