# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:31:02 2024

@author: Leonardo
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Configure o backend aqui
import matplotlib.pyplot as plt



def escalona(l,a,u,b):
    
    prm = [l,a,u,b]
    
    for k in prm:
        assert (type(k) == np.ndarray) , "Tipo ou dimensionalidade inválida para parâmetro de entrada"
        assert (len(k.shape)  == 1) , "Tipo ou dimensionalidade inválida para parâmetro de entrada"

    assert ( ( l.size == a.size - 1 ) and ( u.size == a.size - 1 )  and ( b.size == a.size  ) ), "Quantidade de elementos inválida para sistema tridiagonal"
    
    n = a.size
    
    assert ( (u.size == l.size) and (u.size == (a.size - 1)) and (a.size == b.size) ) ,  "escalona: Os arrays não possuem tamanhos compatíveis,"
    
    up = np.zeros( u.shape )
    bp = np.zeros( b.shape )
    
    up[0] = u[0] / a[0]
    bp[0] = b[0] / a[0]

    assert (a[0] != 0), "escalona: Diagonal principal nula, sistema requer pivotamento." 
    
    for i in range(1,n-1):
        assert( a[i] - l[i-1] * up[i-1]  != 0), "escalona: Diagonal principal nula, sistema requer pivotamento." 
        up[i] = u[i] / ( a[i] - l[i-1] * up[i-1] )
        bp[i] = (b[i] - l[i-1]*bp[i-1]) / ( a[i] - l[i-1] * up[i-1] )
       
    i = n-1
    bp[i] = (b[i] - l[i-1]*bp[i-1]) / ( a[i] - l[i-1] * up[i-1] )        
       
    return up,bp
        
        
def subreg(u,b):
 
    n = len(b)
    x = np.zeros( n )
    x[n-1] = b[n-1]
           
    for i in range(n-2,-1,-1):
            x[i] = b[i] - x[i+1] * u[i]
    return x

def sistematdma(l,a,u,b):
    u,b=escalona(l,a,u,b)
    x=subreg(u,b)
    return x



#############Spline3
def coeficientes(x,f):
    n=len(x)-1
    h=np.zeros(n,dtype=np.float64)
    der=np.zeros(n,dtype=np.float64)
    for i in range(0,n):
       h[i]=x[i+1] -x[i]
       der[i]=(f[i+1]-f[i])/h[i]
    l=np.zeros(n,dtype=np.float64)
    u=np.zeros(n,dtype=np.float64)
    dp=np.zeros(n+1,dtype=np.float64)
    b=np.zeros(n+1,dtype=np.float64)
    l[0]=h[0]
    dp[0]=1
    dp[-1]=1 
    dp[-2]=2*(h[n-2]+h[n-1])
    b[-2]=3*(der[n-1]-der[n-2])#aqui
    u[-1]=h[n-1]
    for i in range(1,n-1):
        l[i]=h[i]
        u[i]=h[i]
        dp[i]=2*(h[i-1]+h[i])
        b[i]=3*(der[i]-der[i-1])
    #########soluções
    a=np.zeros(n,dtype=np.float64)
    bcoef=np.zeros(n,dtype=np.float64)
    c=np.zeros(n+1,dtype=np.float64)
    d=np.zeros(n,dtype=np.float64)
    c=sistematdma(l,dp,u,b)
    for i in range(0,n):
        d[i]=(c[i+1]-c[i])/(3*h[i])
        bcoef[i]=der[i]-(1/3)*(c[i+1]+2*c[i])*h[i]
        a[i]=f[i]
    return a,bcoef,c,d
    
def spline3(x,f,z):
    assert(z>=x[0] and z<=x[-1]),"z deve estar entre o intervalo [x0,xn]"
    i=0
    while z>x[i+1]:
        i=i+1
    a,b,c,d=coeficientes(x,f)    
    return a[i]+b[i]*(z-x[i])+c[i]*(z-x[i])**2+d[i]*(z-x[i])**3

x=np.array([3,4.5,7,9],dtype=np.float64)
y=np.array([2.5,1,2.5,0.5],dtype=np.float64)
spline3(x,y,9)

dom=np.linspace(x[0],x[-1],1000)
tam=len(dom)
polspline=np.zeros(dom.shape)
for i in range(0,tam):
    polspline[i]=spline3(x,y,dom[i])

plt.plot(dom,polspline,'g')
plt.plot(x,y,'bo')
plt.show()
