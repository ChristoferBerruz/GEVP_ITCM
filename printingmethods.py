#This is a module for different printing methods and plotting methods
import matplotlib.pyplot as plt
import numpy as np


def printGEVP_withErrors(Eigevalues_avg_errors, to, outf_name):
	f = open(outf_name, "w")
	p, nt, ni = Eigevalues_avg_errors.shape
	with f:
		for t in range(to+1, nt):
			f.write(" %d " % t)
			for i in range(ni):
				f.write("  %.16E  %.16E  " %(Eigevalues_avg_errors[0,t,i], Eigevalues_avg_errors[1,t,i]))
			f.write("\n")


def printITCM(fname, ITCM, avg):
	'''
	This method prints the matrix ITCM to a text file
	:param ITCM: ITCM Matrix
	:param avg: bool stating if ITCM was averaged over nsamp, nproc.
	:return:
	'''
	nt = ni = nj = 0
	if avg:
		nt, ni, nj = ITCM.shape
	else:
		ITCM_Avg = np.mean(ITCM[:,:,:,:,:,0], axis=(0, 1), dtype=np.float64)
		ITCM = ITCM_Avg
		nt, ni, nj = ITCM_Avg.shape
	f = open(fname, "w")
	with f:
		for t in range(nt):
			for i in range(ni):
				for j in range(nj):
					f.write("%4s  %d  %d  %.16E\n" % (str(t),i+1, j+1, ITCM[t,i,j]))
			f.write("\n")