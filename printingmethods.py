#This is a module for different printing methods and plotting methods
import matplotlib.pyplot as plt
import numpy as np


def printGEVP_withErrors(Eigevalues_avg_errors, to, outf_name):
	'''
	Print the results of GEVP including the errors for each eigval.
	:param Eigevalues_avg_errors: array_like, ndarray containing eigvals and eigerrors
	:param to: step at which the weight matrix was stablished.
	:param outf_name: output file name
	:return:
	'''
	f = open(outf_name, "w")
	p, nt, ni = Eigevalues_avg_errors.shape
	f.write("# t, eigval, error\n")
	with f:
		for t in range(to+1, nt):
			f.write(" %d " % t)
			for i in range(ni):
				f.write("  %.16E  " % Eigevalues_avg_errors[0,t,i])
			for i in range(ni):
				f.write("  %.16E  " % Eigevalues_avg_errors[1,t,i])
			f.write("\n")


def printITCM(fname, ITCM, avg, SIGN):
	'''
	This method prints the matrix C dependent on t.
	:param ITCM: ITCM Matrix
	:param avg: bool stating if ITCM was averaged over nsamp, nproc.
	:param SIGN: SIGN strcuture
	:return:
	'''
	nt = ni = nj = 0
	if avg:
		nt, ni, nj = ITCM.shape
	else:
		ITCM_Avg = np.mean(ITCM[:,:,:,:,:,0], axis=(0, 1), dtype=np.float64)
		ITCM = ITCM_Avg
		nt, ni, nj = ITCM_Avg.shape
	ITCM = np.true_divide(ITCM, np.mean(SIGN, axis=(0,1), dtype=np.float64)[0])
	f = open(fname, "w")
	with f:
		for t in range(nt):
			for i in range(ni):
				for j in range(nj):
					f.write("%4s  %d  %d  %.16E\n" % (str(t),i+1, j+1, ITCM[t,i,j]))
			f.write("\n")