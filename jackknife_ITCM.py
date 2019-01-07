import numpy as np

def jackknife_ITCM(ITCM, SIGN):
	nsamp, nproc, nt, ni, nj, _ = ITCM.shape
	ITCM_Real,_ = return_RealImagPart(ITCM) #We only care about the real part
	ITCM_Real_JK = np.zeros((nsamp, nproc, nt, ni, nj), dtype=np.float64)
	for samp in range(nsamp):
		for proc in range(nproc):
			for t in range(nt):
				ITCM_Real_JK[samp, proc, t] = walker(ITCM_Real, SIGN, samp, proc, t, 0)
	return ITCM_Real_JK


def walker(ITCM_Real, SIGN, s,p,t,real_or_imag):
	"""
	#param: s, p, and t are indexes for the current starting point.
			real_or_imag shall be an integer 0, 1 for real or imag part respectively.
	#return: C matrix for current s, p, and t skipping s and p.
	"""
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