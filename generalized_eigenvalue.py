import numpy as np
from scipy import linalg as la
import h5py
import math
import jackknife_ITCM as jk
import printingmethods as printm
import comparisonmethods as compm

def getStructures_fromH5File(hdf5_filename, key):
	'''
	Retrieve a given value for a key in a specified hdf5 file.
	:param hdf5_filename: name of the Hdf5 file.
	:param key: key to be retrieved.
	:return t: array_like, the value of the key.
	FIXME: Note that key need to be parsed, no interaction with console.
	'''
	with h5py.File(hdf5_filename, "r") as f:
		try:
			t = f.get(key).value
			print("Succefully returned structure")
			return t
		except Exception as e:
			print("Error at getStructures_fromH5File: " + str(e))

def GEVP(ITCM, SIGN, to):
	'''
	Generalized Eigen Value Problem for C(t) dependent on samp, proc matrix with SIGN problem
	:param ITCM: array_like, ITCM Matrix
	:param SIGN: array_like, SIGN Matrix
	:param to: step at which the weight matrix is established
	:return Eigvals_with_errors: array_like,  matrix dependent on t>to containing eigvals and eigerrors.
	FIXME: List comprehension approach not yet achieved. Computing time has improved, yet still desirable.
	'''
	ITCM_JK = jk.jackknife_ITCM(ITCM, SIGN)
	nsamp, nproc, nt, ni, nj = ITCM_JK.shape
	Eigenvalues = np.zeros((nsamp, nproc, nt, ni), dtype=np.float64) #contains eigenvalues
	#Calculating eigenvalues per samp, proc, t > to.
	#If t<to default is 0.0
	for samp in range(nsamp):
		for proc in range(nproc):
			for t in range(to+1,nt):
				B = symmetricMatrix(np.array(ITCM_JK[samp, proc, to], dtype=np.float64))
				A = symmetricMatrix(np.array(ITCM_JK[samp, proc, t], dtype=np.float64))
				Eigenvalues[samp, proc, t], _ = la.eigh(A, B, lower=False)
	#Averaging eigenvalues over nsamp, nproc
	Eigenvalues_averaged = np.mean(Eigenvalues, axis=(0,1), dtype=np.float64)
	#Calculating jackknife errors for t > to
	Eigenvalues_errors = np.zeros((nt, ni), dtype=np.float64)
	for t in range(to+1, nt):
		dummy = np.zeros(ni, dtype=np.float64)
		for samp in range(nproc):
			for proc in range(nproc):
				dummy = np.add(dummy, (Eigenvalues[samp, proc, t]-Eigenvalues_averaged[t])**2)
		Eigenvalues_errors[t] = np.true_divide(np.multiply(nsamp*nproc-1, dummy), (nsamp*nproc))
		Eigenvalues_errors[t] = np.sqrt(Eigenvalues_errors[t])
	return np.array([Eigenvalues_averaged[:, :],Eigenvalues_errors[:,:]],dtype=np.float64)

def symmetricMatrix(A):
	'''
	This method forces a matrix to be symmetric.
	:param A: 2-D matrix
	:return S_symm: symmetric version of A copying the upper triangular matrix
	'''
	ni, nj = A.shape
	l = 1
	A_symm = np.triu(A)
	for i in range(ni):
		for j in range(l, nj):
			A_symm[j,i] = A_symm[i,j]
		l += 1
	return A_symm

def solveGEV(hdf5filename, to):
	'''
	Driver to solve GEVP.
	:param hdf5filename: HDF5 file to solve.
	:param to: step at which weight matrix is selected.
	:return:
	FIXME: Files are generated properly. Global integration still on the work.
	'''
	filename = hdf5filename
	ITCM = getStructures_fromH5File(filename, 'C(K=2,pi=+)')
	SIGN = getStructures_fromH5File(filename, 'sign')
	Eigvals_and_errors = GEVP(ITCM, SIGN, to)
	outfile_gevp = filename + ".GEVP.txt"
	outfile_itcm = filename + ".ITCM.txt"
	printm.printGEVP_withErrors(Eigvals_and_errors, to, outfile_gevp)
	printm.printITCM(outfile_itcm, ITCM, avg=False, SIGN=SIGN)


def main():
	filename = "10Be.db32.nb016.h5.A"
	to = 0
	solveGEV(filename,to)
	outfile_GEVP = filename + ".GEVP.txt"
	comparisonfile = "test.lamb.NoChange"
	compm.GEVP(outfile_GEVP,comparisonfile,to, savePDF=False, filename=filename)


if __name__ == '__main__':main()