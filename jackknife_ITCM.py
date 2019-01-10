import numpy as np


def jackknife_ITCM(ITCM, SIGN):
	nsamp, nproc, nt, ni, nj, _ = ITCM.shape
	ITCM_Real,_ = return_RealImagPart(ITCM) #We only care about the real part
	ITCM_Real_JK = np.zeros((nsamp, nproc, nt, ni, nj), dtype=np.float64)
	for t in range(nt):
		for samp in range(nsamp):
			for proc in range(nproc):
				ITCM_Real_JK[samp, proc, t] = walker(ITCM_Real, SIGN, samp, proc, t, 0)
	return ITCM_Real_JK


def walker(ITCM_Real, SIGN, s,p,t,real_or_imag):
	'''
	Iterates over nsamp, nproc skipping current s, p (samp, proc)
	:param ITCM_Real: Real or Imag part only of ITCM
	:param SIGN: SIGN structure
	:param s: current samp
	:param p: current proc
	:param t: current tau
	:param real_or_imag: (int) real or imag part of SIGN. Must be between 0-1
	:return sum(C(t,i)/SIGN(i)) for i != (s,p): array_like, 2D Array
	FIXME: List comprehension approach desirable. Computation time not efficient enough.
	'''
	nsamp, nproc, nt, ni, nj = ITCM_Real.shape
	dummy_c = np.zeros((ni, nj), dtype=float)
	dummy_s = 0
	for samp in range(nsamp):
		for proc in range(nproc):
			if (samp == s and proc == p):
				#We skip the current position
				continue
			dummy_c = np.add(dummy_c, ITCM_Real[samp][proc][t])
			dummy_s += SIGN[samp][proc][real_or_imag]
	return np.true_divide(dummy_c, dummy_s)
	

def return_RealImagPart(ITCM):
	'''
	:param ITCM: Matrix ITCM
	:return: an array containing, separately, the Real Part and Imag Part of ITCM
	'''
	ITCM_Real = np.array(ITCM[:, :, :, :, :, 0], dtype=np.float64)
	ITCM_Imag = np.array(ITCM[:, :, :, :, :, 1], dtype=np.float64)
	return [ITCM_Real, ITCM_Imag]

def jackknife_statistics(A, SIGN, real_or_imag, resampled_set_only=True):
	'''
	Perform jackknife statistics in A, SIGN with dimensions (nsamp, nproc, 2)
	:param A: The observable to be studied.
	:param SIGN: sign
	:param real_or_imag: real or imaginary part of the calculation. 0 = real, 1 = imag
	:param resampled_set_only: bool. If True, it only return the jackknife resampled data set.
									 Otherwise, it returns [dataset, jk_av, error]
	:return:
	'''
	nsamp, nproc, _ = A.shape
	if A.shape != SIGN.shape:
		raise Exception("Shapes do not match. Jackknife is not possible.")
	M = nsamp*nproc
	A = np.array([A[samp,proc,real_or_imag] for samp in range(nsamp) for proc in range(nproc)], \
				 dtype=np.float64)
	print(A.shape)
	SIGN = np.array([SIGN[samp,proc,real_or_imag] for samp in range(nsamp) for proc in range(nproc)], \
				 dtype=np.float64)
	A_jk = np.zeros(nsamp*nproc, dtype=np.float64)
	#Jackknife occures in the following loop.
	for i in range(M):
		A_jk[i] = np.sum(np.array([A[j] for j in range(M) if i!=j]), dtype=np.float64)
		d = np.sum(np.array([SIGN[j] for j in range(M) if i!= j]), dtype=np.float64)
		A_jk[i] = np.true_divide(A_jk[i], d)
	if resampled_set_only:
		return A_jk
	else:
		jk_avg = np.mean(A_jk)
		error = np.sum(np.array([(jk_avg-a)**2 for a in A_jk], dtype=np.float64))
		error = np.sqrt((M-1)*error/M)
		return[A_jk, jk_avg, error]
