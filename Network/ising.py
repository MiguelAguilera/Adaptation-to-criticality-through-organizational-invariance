import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt


class ising:
	def __init__(self, netsize):  # Create ising model

		self.size = netsize
		self.h = np.zeros(netsize)
		self.J = np.zeros((netsize, netsize))
		self.randomize_state()
		self.Beta = 1.0

	def randomize_state(self):
		self.s = np.random.randint(0, 2, self.size) * 2 - 1

	def pdf(self):  # Get probability density function of ising model with parameters h, J

		self.P = np.zeros(2**self.size)
		for n in range(2**self.size):
			s = bitfield(n, self.size) * 2 - 1
#			self.P[n]=np.exp(self.Beta*(np.dot(s,self.h) + np.dot(np.dot(s,self.J),s)))
			P1 = np.exp(self.Beta * (np.dot(s, self.h) + np.dot(np.dot(s, self.J), s)))
			self.P.itemset(n, P1)
		self.Z = np.sum(self.P)
		self.P /= self.Z

	def random_wiring(self):  # Set random values for h and J
		self.h = np.random.randn(self.size)
		self.J = np.zeros((self.size, self.size))
		for i in np.arange(self.size):
			for j in np.arange(i + 1, self.size):
				self.J[i, j] = np.random.randn(1)

	def random_rewire(self):
		if np.random.rand(1) > 0.5:
			self.h[np.random.randint(self.size)] = np.random.randn(1)
		else:
			i = np.random.randint(self.size - 1)
			j = np.random.randint(i, self.size)
			self.J[i, j] = np.random.randn(1)

	def independent_model(self, m):  # Set h to match an independen models with means m
		self.h = np.zeros((self.size))
		for i in range(self.size):
			self.h[i] = -0.5 * np.log((1 - m[i]) / (1 + m[i]))
		self.J = np.zeros((self.size, self.size))

	def observables(self):  # Get mean and correlations from probability density function
		self.pdf()
		self.m = np.zeros((self.size))
		self.c = np.zeros((self.size, self.size))
		self.C = np.zeros((self.size, self.size))
		for n in range(2**self.size):
			s = bitfield(n, self.size) * 2 - 1
			self.m += self.P[n] * s
			for i in range(self.size):
				self.c[i, i + 1:] += self.P[n] * s[i] * s[i + 1:]
		for i in range(self.size):
			self.C[i, i + 1:] = self.c[i, i + 1:] - self.m[i] * self.m[i + 1:]

	def observablesMC(self, T):  # Get mean and correlations from MonteCarlo samples
		self.m = np.zeros((self.size))
		self.c = np.zeros((self.size, self.size))
		self.C = np.zeros((self.size, self.size))
		for t in range(T):
			self.SequentialGlauberStep()
			self.m += self.s
#			self.c+=np.triu(np.tensordot(self.s,self.s,axes=0),1)
			for i in range(self.size):
				self.c[i, i + 1:] += self.s[i] * self.s[i + 1:]
		self.m /= T
		self.c /= T
		for i in range(self.size):
			self.C[i, i + 1:] = self.c[i, i + 1:] - self.m[i] * self.m[i + 1:]

	# Solve exact inverse ising problem with gradient descent
	def inverse_exact(self, m1, C1, error):
		u = 0.1
		count = 0
		self.independent_model(m1)

		self.observables()
		fit = max(np.max(np.abs(self.m - m1)), np.max(np.abs(self.C - C1)))
		fmin = fit

		while fit > error:

			dh = u * (m1 - self.m)
			self.h += dh
			dJ = u * (C1 - self.C)
			self.J += dJ

			self.observables()
			fit = max(np.max(np.abs(self.m - m1)), np.max(np.abs(self.C - C1)))
			count += 1
			if count % 10 == 0:
				print(self.size, count, fit)

		return fit

	def MetropolisStep(self, i=None):  # Execute step of Metropolis algorithm
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if eDiff <= 0 or np.random.rand() < np.exp(-self.Beta * eDiff):    # Metropolis
			self.s[i] = -self.s[i]

	# Execute step of Metropolis algorithm with zero temperature (deterministic)
	def MetropolisStepT0(self, i=None):
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if eDiff <= 0:
			self.s[i] = -self.s[i]

	def GlauberStep(self, i=None):  # Execute step of Glauber algorithm
		if i is None:
			i = np.random.randint(self.size)
		eDiff = 2 * self.s[i] * \
			(self.h[i] + np.dot(self.J[i, :] + self.J[:, i], self.s))
		if eDiff < np.log(1 / np.random.rand() - 1) / self.Beta:    # Glauber
			self.s[i] = -self.s[i]

	def SequentialGlauberStep(self):
		for i in np.random.permutation(self.size):
			self.GlauberStep(i)

	def deltaE(self, i):  # Compute energy difference between two states with a flip of spin i
		return 2 * (self.s[i] * self.h[i] + np.sum(self.s[i] * \
		            (self.J[i, :] * self.s) + self.s[i] * (self.J[:, i] * self.s)))

	def metastable_states(self):  # Find the metastable states of the system
		self.pdf()
		ms = []
		Pms = []
		for n in range(2**self.size):
			m = 1
			s = bitfield(n, self.size)
			for i in range(self.size):
				s1 = s.copy()
				s1[i] = 1 - s1[i]
				n1 = bool2int(s1)
				if self.P[n] < self.P[n1]:
					m = 0
					break
			if m == 1:
				ms += [n]
				Pms += [self.P[n]]
		return ms, Pms

	def get_valley(self, s):  # Find an attractor "valley" starting from state s
		ms, Pms = self.metastable_states()
		n = bool2int((s + 1) / 2)
		self.s = s.copy()
		while n not in ms:
			self.MetropolisStepT0()
			n = bool2int((self.s + 1) / 2)
		ind = ms.index(n)
		valley = ind
		print(ind, n, ms)
		return valley

	def energy(self):  # Compute energy function
		self.pdf()
		self.E = np.zeros(2**self.size)
		for n in range(2**self.size):
			s = bitfield(n, self.size) * 2 - 1
			self.E[n] = -(np.dot(s, self.h) + np.dot(np.dot(s, self.J), s))
		self.Em = np.sum(self.P * self.E)

	def HeatCapacity(self):  # Compute energy function
		self.HC = self.Beta**2 * \
			(np.sum(self.P * self.E**2) - np.sum(self.P * self.E)**2)

	def MutualInformation(self, rngx, rngy):
		rngxy = rngx + rngy
		Pxy = subPDF(self.P, rngxy)
		rngx1 = np.arange(len(rngx))
		rngy1 = np.arange(len(rngx), len(rngx) + len(rngy))
		return(MI(Pxy, rngx1, rngy1))


def bool2int(x):  # Transform bool array into positive integer
    y = 0
    for i, j in enumerate(np.array(x)[::-1]):
        y += j * 2**i
    return y


def bitfield(n, size):  # Transform positive integer into bit array
    x = [int(x) for x in bin(n)[2:]]
    x = [0] * (size - len(x)) + x
    return np.array(x)


def subPDF(P, rng):
	subsize = len(rng)
	Ps = np.zeros(2**subsize)
	size = int(np.log2(len(P)))
	for n in range(len(P)):
		s = bitfield(n, size)
		Ps[bool2int(s[rng])] += P[n]
	return Ps


def Entropy(P):
	E = 0.0
	for n in range(len(P)):
		if P[n] > 0:
			E += -P[n] * np.log(P[n])
	return E


def MI(Pxy, rngx, rngy):
	size = int(np.log2(len(Pxy)))
	Px = subPDF(Pxy, rngx)
	Py = subPDF(Pxy, rngy)
	I = 0.0
	for n in range(len(Pxy)):
		s = bitfield(n, size)
		if Pxy[n] > 0:
			I += Pxy[n] * np.log(Pxy[n] / (Px[bool2int(s[rngx])]
                                  * Py[bool2int(s[rngy])]))
	return I


def TSE(P):
	size = int(np.log2(len(P)))
	C = 0
	for npart in np.arange(1, 0.5 + size / 2.0).astype(int):
		bipartitions = list(combinations(range(size), npart))
		for bp in bipartitions:
			bp1 = list(bp)
			bp2 = list(set(range(size)) - set(bp))
			C += MI(P, bp1, bp2) / float(len(bipartitions))
	return C


def KL(P, Q):
	D = 0
	for i in range(len(P)):
		D += P[i] * np.log(P[i] / Q[i])
	return D


def JSD(P, Q):
	return 0.5 * (KL(P, Q) + KL(Q, P))
