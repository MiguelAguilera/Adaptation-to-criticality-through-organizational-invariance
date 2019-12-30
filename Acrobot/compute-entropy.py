#!/usr/bin/env python

from embodied_ising import ising
import numpy as np
import matplotlib.pyplot as plt
from info_theory import Entropy
from sys import argv

if len(argv) < 3:
    print("Usage: " + argv[0] + " <N> + <bind>")
    exit(1)

N=int(argv[1])
bind=int(argv[2])

size=6*N
Nsensors=2*N
Nmotors=N

R=10

Nbetas=101
betas=10**np.linspace(-1,1,Nbetas)
Ha=np.zeros(R)
Hn=np.zeros(R)
Hs=np.zeros(R)
Hp=np.zeros(R)

Iterations=1000
T=5000


for ind in range(R):
	print()
	print(betas[bind],size,mode)
	
	I=ising(size,Nsensors,Nmotors)
	filename='files/network-size_'+str(size)+'-sensors_'+str(Nsensors)+'-motors_'+str(Nmotors)+'-T_'+str(T)+'-Iterations_'+str(Iterations)+'-ind_'+str(ind)+'.npz'

	data=np.load(filename)
	I.h=data['h']
	I.J=data['J']

	beta=betas[bind]
	I.Beta=betas[bind]
	T1=100000*size
	s=np.zeros(T1)
	m=np.zeros(T1)
	h=np.zeros(T1)
	a=np.zeros(T1)
	t=0
	I.randomize_position()
	T0=int(T1/10)
	for t0 in range(T0):
		I.SequentialUpdate()
	
	F=0
	for t in range(T1):
		
		I.SequentialUpdate()
		s[t]=I.get_state_index('sensors')
		m[t]=I.get_state_index('motors')
		h[t]=I.get_state_index('non-sensors')
		a[t]=I.get_state_index()
		Hi= I.h + np.dot(I.s,I.J)+ np.dot(I.J,I.s)
		Fi=beta*Hi*np.tanh(beta*Hi)-np.log(2*np.cosh(beta*Hi))
		F+=np.sum(Fi[Nsensors:])
	F/=T1
	Hp[ind]=F
		
	Ha[ind]=Entropy(a)
	Hs[ind]=Entropy(s)
	Hn[ind]=Entropy(h)

		

	print('Entropy agent',Ha[ind],'Entropy sensor',Hs[ind],'Entropy sensor',Hn[ind])
	
filename='H/network-size_'+str(size)+'-sensors_'+str(Nsensors)+'-motors_'+str(Nmotors)+'-T_'+str(T)+'-Iterations_'+str(Iterations)+'-bind_'+str(bind)+'.npz'
np.savez(filename,betas=betas,Nbetas=Nbetas,Ha=Ha,Hn=Hn,Hs=Hs,Hp=Hp)

