#!/usr/bin/env python

from embodied_ising import ising,bool2int
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

plt.rc('text', usetex=True)
font = {'family':'serif','size':12, 'serif': ['computer modern roman']}
plt.rc('font',**font)
plt.rc('legend',**{'fontsize':16})

N=64

Nsensors=4
Nmotors=2
size=N+Nsensors+Nmotors

ind=0

#beta=1.2
beta=1
Iterations=1000
T=5000
visualize=True
# visualize=False


I=ising(size,Nsensors,Nmotors)
I.Beta=beta



filename='files/network-size_'+str(size)+'-sensors_'+str(Nsensors)+'-motors_'+str(Nmotors)+'-T_'+str(T)+'-Iterations_'+str(Iterations)+'-ind_'+str(ind)+'.npz'

data=np.load(filename)
I.h=data['h']
I.J=data['J']

plt.figure()
plt.bar(range(size),I.h)
plt.figure()
plt.imshow(I.J,interpolation='nearest')
plt.colorbar()

T=4000*2
if beta>1:
	T*=5
p=np.zeros(T)
s=np.zeros(T)
m=np.zeros(T)
h=np.zeros(T)
a=np.zeros(T)
n=np.zeros(T,int)
spins=np.zeros((size,T))
acc=np.zeros(T)
spd=np.zeros(T)
pos=np.zeros(T)
height=np.zeros(T)

nsize=size
I.env.reset()
T0=10000

for t in range(T0):
	I.SequentialUpdate()
for t in range(T):
	I.SequentialUpdate()
	s[t]=I.get_state_index('input')
#	a[t]=I.get_state_index('non-sensors')
	h[t]=I.get_state_index('hidden')
#	m[t]=I.get_state_index('motors')
	acc[t]=I.acceleration
	spd[t]=I.speed
	pos[t]=I.env.state[0]
	height[t]=I.height
		
			
	if visualize:
		I.env.render()

plt.figure()
plt.plot(h)
plt.title('hidden units')
plt.figure()
plt.plot(s)
plt.title('inputs')



if beta==1:
	letter='b'
elif beta<1:
	letter='a'
else:
	letter='c'
fig, ax = plt.subplots(1,1,figsize=(4.6,3.8))
plt.rc('text', usetex=True)
plt.plot(pos,spd,'k')
plt.ylabel(r'$v$',fontsize=18, rotation=0)
plt.xlabel(r'$x$',fontsize=18)
plt.title(r'$\beta='+str(beta)+'$',fontsize=36)
plt.axis([-np.pi/2-0.05,np.pi/6+0.05,-0.05,0.05])
plt.savefig('img/fig6'+letter+'.pdf',bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(4,2))
plt.rc('text', usetex=True)
plt.plot(pos,'k')
plt.ylabel(r'$x$',fontsize=18, rotation=0)
plt.xlabel(r'$t$',fontsize=18)
plt.axis([0,len(pos),-np.pi/2-0.05,np.pi/6+0.05])
plt.savefig('img/fig6'+letter+'1.pdf',bbox_inches='tight')



plt.show()

