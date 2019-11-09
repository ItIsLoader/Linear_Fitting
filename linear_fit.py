import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def algebraic_relation(a,b,c,d,e,f):
    m = np.array([[1, 0, ((d*e-b*f)/(a*d-b*c))],[0, 1, ((a*f-c*e)/(a*d-b*c))]])
    pos_a = m[0,-1]
    pos_b = m[1,-1]
    return np.array([[pos_a],[pos_b]])

def linear_fit(x, y, yerr):
    sum1 = np.sum(1/(yerr**2))
    sumx = np.sum(x/(yerr**2))
    sumy = np.sum(y/(yerr**2))
    sumxy = np.sum(x*y/(yerr**2))
    sumx2 = np.sum((x**2)/(yerr**2))
    delta = (sum1*sumx2) - ((sumx)**2)
    fit_erra = np.sqrt((1/delta)*sumx2)
    fit_errb = np.sqrt((1/delta)*sum1)
    chi2 = np.sum(((y - pos_a - (pos_b*x))/yerr)**2)
    dof = len(x) - 1
    
    return (algebraic_relation(sum1, sumx, sumx, sumx2, sumy, sumxy), np.array[fit_erra,fit_errb], chi2, dof)

def fitted_data(x, y, yerr):
    y_int = linear_fit(x, y, yerr)[0] #This is the value of a
    slope = linear_fit(x, y, yerr)[1] #This is the value of b
    
    plt.figure(1)
    plt.errorbar(x, y, yerr=yerr, xerr=None, fmt='bo', ecolor='r', label='Data w/ Error Bars')
    plt.plot(x, slope*x + y_int, label='Linear Fit')
    plt.title('Linear Fit on data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend
    plt.show()
    
    plt.figure(2)
    residual = ((y - y_int - (slope*x)) / (yerr))
    plt.plot(x, residual, 'o')
    plt.title('Residuals vs. x')
    plt.xlabel('x')
    plt.ylabel('Residuals')
    plt.show()
    
    chi2 = np.sum(((y - y_int - (slope*x))/(yerr))**2)
    dof = len(x)-1
    p = 1-stats.chi2.cdf(5.0**2,1)
    print('chi2 equals '+str(chi2))
    print('dof equals '+str(dof))
    print('probability level equals '+str(p))