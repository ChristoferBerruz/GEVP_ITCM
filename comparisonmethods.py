import numpy as np
import matplotlib.pyplot as plt

def GEVP(f1_name, f2_name, to, savePDF=False):
    '''
    Comparison results of a GEVP. Same format is assumed (or at least some exception are managed)
    :param f1_name: One of the files for comparison
    :param f2_name: The other file for comparison.
    :param savePDF:
    :return:
    '''
    to += 1
    pos = []
    eigvals = []
    errors = []
    t = []
    f1 = open(f1_name, "r")
    f2 = open(f2_name, "r")
    with f1:
        with f2:
            for line1, line2 in zip(f1,f2):
                if len(line1.split())==13 and len(line2.split())==13:
                    cline1 = line1.split()
                    cline2 = line2.split()
                    for s in range(13):
                        if s == 0:
                            pos.append(abs(float(cline1[s]) - float(cline2[s])))
                        elif s < 7:
                            eigvals.append(abs(float(cline1[s]) - float(cline2[s])))
                            t.append(to)
                        else:
                            errors.append(abs(float(cline1[s]) - float(cline2[s])))
                    to +=1
    plt.plot(t, eigvals, 'o',markersize=1.50)
    plt.title("Eigvals")
    plt.show()