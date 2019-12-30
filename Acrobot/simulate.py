#!/usr/bin/env python

from embodied_ising import ising,bool2int
import numpy as np
import matplotlib.pyplot as plt
from infoflow import MI,Imin,TE,TE1,Entropy,ConditionalEntropy
import time
from scipy.interpolate import interp1d


plt.rc('text', usetex=True)
font = {'family':'serif','size':15, 'serif': ['computer modern roman']}
plt.rc('font',**font)
plt.rc('legend',**{'fontsize':16})


N=64
Nsensors=4
Nmotors=2
size=N+Nsensors+Nmotors

ind=0


beta=1.0
#beta=1.2

Iterations=1000
T=5000
visualize=True
visualize=False



filename='files/network-size_'+str(size)+'-sensors_'+str(Nsensors)+'-motors_'+str(Nmotors)+'-T_'+str(T)+'-Iterations_'+str(Iterations)+'-ind_'+str(ind)+'.npz'

I=ising(size,Nsensors,Nmotors)
I.Beta=beta
data=np.load(filename)
I.h=data['h']
I.J=data['J']

plt.figure()
plt.bar(range(size),I.h)
plt.figure()
plt.imshow(I.J,interpolation='nearest')
plt.colorbar()




T=10000
si=np.zeros(T)
s=np.zeros(T)
m=np.zeros(T)
h=np.zeros(T)
spd=np.zeros(T)
xpos=np.zeros(T)
ypos=np.zeros(T)
theta=np.zeros(T)
P={}

I.randomize_state()
I.env.reset()

T0=10000

for t in range(T0):
	I.SequentialUpdate()
for t in range(T):
	I.SequentialUpdate()
	si[t]=I.get_state_index('input')
	s[t]=I.get_state_index('sensors')
	h[t]=I.get_state_index('hidden')
	m[t]=I.get_state_index('motors')
	spd[t]=I.speed
	theta[t]=I.theta
	xpos[t]=I.xpos
	ypos[t]=I.ypos
	n=I.get_state_index()
	
	if not n in P:
		P[n]=1
	else:
		P[n]+=1
	t+=1
	if visualize:
		I.env.render()
		time.sleep(0.01) 
	
print(Entropy(si)/4,Entropy(s)/4, Entropy(h)/N)
plt.figure()
plt.plot(h)
plt.figure()
plt.plot(si)
plt.figure()
plt.plot(spd/I.maxspeed)

plt.figure()
plt.plot(ypos)

P=list(P.values())
P/=np.sum(P)

order=np.argsort(P)[::-1]
r=np.arange(1,len(P)+1)
plt.figure()
plt.loglog(r,P[order])
Psf = 1.0/(1+np.arange(len(P)))
Psf/=np.sum(Psf)
plt.loglog(r,Psf,'--g')

if beta==1:
	letter='e'
elif beta<1:
	letter='d'
else:
	letter='f'
	
dT=60
T1=(2*dT+1)*10

t0=np.argmax(ypos)
t=np.arange(0,1+2*dT)
x=xpos[t0-dT:t0+dT+1]
y=ypos[t0-dT:t0+dT+1]
print(len(t),len(x),len(y))
fy = interp1d(t, y, kind='cubic')
fx = interp1d(t, x, kind='cubic')
t1 = np.linspace(0, 2*dT, T1)
x1=fx(t1)
y1=fy(t1)


fig, ax = plt.subplots(1,1,figsize=(4.6,3.8))
plt.rc('text', usetex=True)
plt.plot(x1,y1,'k')
plt.ylabel(r'$y$',fontsize=18, rotation=0)
plt.xlabel(r'$x$',fontsize=18)
plt.title(r'$\beta='+str(beta)+'$',fontsize=36)
plt.axis([-2,2,-2,2])
plt.savefig('img/fig6'+letter+'.pdf',bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(4,2))
plt.rc('text', usetex=True)
plt.plot(ypos,'k')
plt.ylabel(r'$y$',fontsize=18, rotation=0)
plt.xlabel(r'$t$',fontsize=18)
plt.axis([0,T,-2,2])
plt.savefig('img/fig6'+letter+'1.pdf',bbox_inches='tight')
plt.show()
