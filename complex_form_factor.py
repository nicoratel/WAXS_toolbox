import xraydb
import numpy as np
import functools
import requests
from io import StringIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import math
import h5py #format qui contient tous les fichiers: celui de q et ceux de I

def _f0(qmin,qmax,qstep,element):
    qarray=np.arange(qmin,qmax,qstep)
    f=xraydb.f0(element,qarray)
    return f

def atomicformfactor_nist(element,wavelength):
    Z=xraydb.atomic_number(element)

    if Z<3:
        urltmplate = 'https://physics.nist.gov/cgi-bin/ffast/ffast.pl?gtype=4&Z={Z}'
        url = urltmplate.format(Z=Z)
        session = requests.Session()
        r = session.get(url).content
        # parse page
        soup = BeautifulSoup(r,features="lxml")
        text=soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        index=text.find('fNT =')
        start=index+len('fNT = ')
        fNT=float(text[start:start+11])
        index=text.find('frel (H82,3/5CL) = ')
        start=index+len('frel (H82,3/5CL) = ')
        frel=float(text[start:start+11])
        tabledata = soup.select('pre')[0].text.splitlines()[3:]
        #tabledata = soup.select('pre')[1].text.splitlines()[3:]
        #print(tabledata)
        tabledata = '\n'.join(tabledata)
        data = np.genfromtxt(StringIO(tabledata))
        ev = data[:, 0] * 1e3
        #print('ev',ev)
        f = data[:,1] + 1j*data[:,2]
        #print('f',f)
        f1_array=f.real
        f2_array=f.imag
        energy=12314/wavelength
        # Interpolation à l'énergie souhaitée
        f1=np.interp(energy,ev,f1_array)
        f2=np.interp(energy,ev,f2_array)        
        
    if Z<92 and Z>3:
        #baseurl = 'https://physics.nist.gov/cgi-bin'
        urltmplate = 'https://physics.nist.gov/cgi-bin/ffast/ffast.pl?gtype=4&Z={Z}'
        url = urltmplate.format(Z=Z)
        session = requests.Session()
        r = session.get(url).content
        # parse page
        soup = BeautifulSoup(r,features="lxml")
        text=soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        index=text.find('fNT =')
        start=index+len('fNT = ')
        fNT=float(text[start:start+11])
        index=text.find('frel (H82,3/5CL) = ')
        start=index+len('frel (H82,3/5CL) = ')
        frel=float(text[start:start+11])
        #tabledata = soup.select('pre')[0].text.splitlines()[3:]
        tabledata = soup.select('pre')[1].text.splitlines()[3:]
        #print(tabledata)
        tabledata = '\n'.join(tabledata)
        data = np.genfromtxt(StringIO(tabledata))
        ev = data[:, 0] * 1e3
        #print('ev',ev)
        f = data[:,1] + 1j*data[:,2]
        #print('f',f)
        f1_array=f.real
        f2_array=f.imag
        energy=12314/wavelength
        # Interpolation à l'énergie souhaitée
        f1=np.interp(energy,ev,f1_array)
        f2=np.interp(energy,ev,f2_array)
    
    return f1, fNT, frel, f2
def _f_complex(element,qmin,qmax,qstep,wavelength):
    Z=xraydb.atomic_number(element)
    # définition du tableau de valeurs de q
    qarray=np.arange(qmin,qmax,qstep)
    # Calcul de f0
    f0=_f0(qmin,qmax,qstep,element)
    # Calcul de la composante dispersive f1+frel+fNT
    fdisp=atomicformfactor_nist(element,wavelength)
    f1=fdisp[0]
    fNT=fdisp[1]
    frel=fdisp[2]
    f2=fdisp[3]
    
    # Calcul de f sous forme complexe
    f=np.zeros(len(qarray),dtype=np.complex_)
    real=f0+f1+frel+fNT-Z
    imag=np.full_like(f0,f2)
    f=real+imag*1j
    #print('f_complex for element %s'%str(element)+'=',f)
    return f

