from PyQt5 import QtWidgets,QtCore,QtGui
import sys
import os
import webbrowser
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from gui_new import Ui_WAXS_toolbox
from pathlib import Path
from ase.cluster import Icosahedron,Decahedron,Octahedron
from diffpy.structure import loadStructure
from diffpy.srreal.pdfcalculator import PDFCalculator, DebyePDFCalculator
from diffpy.srreal.parallel import createParallelCalculator
from diffpy.structure.parsers import getParser
from diffpy.srfit.fitbase import FitContribution, FitRecipe
from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.pdf import PDFParser, DebyePDFGenerator, PDFGenerator
from diffpy.structure import Structure
from diffpy.structure.expansion.makeellipsoid import makeSphere, makeEllipsoid
from scipy.optimize import least_squares
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.srfit.pdf.characteristicfunctions import sphericalCF
import multiprocessing
import h5py
import hdf5plugin
class EmittingStream(QtCore.QObject):

    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))
        
    # def flush(self):
    #     sys.stdout.flush()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        
        self.ui = Ui_WAXS_toolbox()
        self.ui.setupUi(self)
        # initialize stdout
        
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        
        
        # Set Working directory
        self.ui.actionSet_Path.triggered.connect(self.getWD)
        self.path=os.getcwd()
        # plot
        self.ui.actionplot.triggered.connect(self.plotlinear)
        # ASE help
        self.ui.actionHelp.triggered.connect(self.ASEhelp)
        
        # ASE Output directory
        self.ui.Set_Path_pushButton.clicked.connect(self.outputspec)
        self.ui.Set_Path2_pushButton.clicked.connect(self.outputspec2)
        # Icosahedron generation
        self.ui.Generate_Ico_pushButton.clicked.connect(self.ico)
        
        # Decahedron generation
        self.ui.Generate_Deca_pushButton.clicked.connect(self.deca)
        
        # Octahedron generation
        self.ui.Generate_Octa_pushButton.clicked.connect(self.octa)
        
        # View structure with Jmol
        self.ui.action_Jmol.triggered.connect(self.jmol)
        
        # cluster from cif
        self.ui.cifpath_pushButton.clicked.connect(self.openciffile)
        self.ui.MakeXYZ_pushButton.clicked.connect(self.makexyz)
        
              
        
        # PDF simulation
        self.ui.simu_qdamp_lineEdit.setText('0.013')
        self.ui.simu_qbroad_lineEdit.setText('0.023')
        self.ui.simu_qmin_lineEdit.setText('0.01')
        self.ui.simu_qmax_lineEdit.setText('25')
        self.ui.simu_rmin_lineEdit.setText('0')
        self.ui.simu_rmax_lineEdit.setText('25') 
        self.ui.simu_rstep_lineEdit.setText('0.02')
        self.ui.simu_Stru_File_pushButton.clicked.connect(self.open_simu_stru)
        self.ui.simu_Plot_Save_pushButton.clicked.connect(self.pdfcalc)
        self.ui.Uij_lineEdit.setText('0.005')
        self.ui.wavelength_lineEdit.setText('0.7107')
        self.ui.iq_qmin_lineEdit.setText('4')
        self.ui.iq_qmax_lineEdit.setText('16.5')
        self.ui.iq_qstep_lineEdit.setText('0.02')
        
        
        # PDF fitting
        self.RUN_PARALLEL=True
        self.ui.qdamp_lineEdit.setText('0.013')
        self.ui.qbroad_lineEdit.setText('0.023')
        self.ui.qmin_lineEdit.setText('0.01')
        self.ui.qmax_lineEdit.setText('16.6')
        self.ui.rmin_lineEdit.setText('0')
        self.ui.rmax_lineEdit.setText('50') 
        self.ui.rstep_lineEdit.setText('0.02')
        self.ui.scaleFactor_lineEdit.setText('0.1')
        self.ui.delta2_lineEdit.setText('4')
        self.ui.sphericalCF_checkBox.setChecked(False)
        self.ui.fitID_lineEdit.setText('Fit_name')
        self.ui.BrowseExpFilepushButton.clicked.connect(self.open_exp_pdf)
        self.ui.BrowseCifFilepushButton.clicked.connect(self.open_fit_strufile)
        self.ui.RunRefinementpushButton.clicked.connect(self.rundiffpy)  
        
        return
    
    
    def __del__(self):
    # Restore sys.stdout
        sys.stdout = sys.__stdout__

    def normalOutputWritten(self, text):
        self.ui.Console_textEdit.append(text)
        return
        
    def getWD(self):
        self.path=QtWidgets.QFileDialog.getExistingDirectory()
        self.ui.Console_textEdit.insertPlainText("Working directory was set to %s" %self.path + "\n")
        self.ui.Path_lineEdit.setText("%s"%self.path)
        self.ui.Path2_lineEdit.setText("%s"%self.path)
        self.outputpath=self.path
        return self.path
    
    def ASEhelp(self):
        webbrowser.open("https://wiki.fysik.dtu.dk/ase/ase/cluster/cluster.html")
        return
    
    def plotlinear(self):
        
        filelist=QtWidgets.QFileDialog.getOpenFileNames( self,"Open 1D data files","%s"%self.path)
        filenames=filelist[0]
        
        plt.figure(1)
        plt.xlabel(r'r ( $\mathrm{\AA}$ )')
        plt.ylabel(r'G(r)')
        i=0
        for file in filenames:
            self.ui.Console_textEdit.insertPlainText(file)
            filename, extension = os.path.splitext(str(file))
            
            label=Path(file).stem
            if extension=='.gr':
                q,i=np.loadtxt(str(file),unpack=True,skiprows=27)
            else:                
                q,i=np.loadtxt(str(file),unpack=True,skiprows=2)
        
            plt.plot(q,i,label=label)
            i+=1
        plt.legend()
        plt.show()
        return    
    
    def ico_size(self,Icosahedron):
        xyz_coord=Icosahedron.get_positions()
        x=list(zip(*xyz_coord))[0];y=list(zip(*xyz_coord))[1];z=list(zip(*xyz_coord))[2]
        x_center=np.mean(x);y_center=np.mean(y);z_center=np.mean(z)
        x_ok=x-x_center;y_ok=y-y_center;z_ok=z-z_center
        r=(x_ok**2+y_ok**2+z_ok**2)**(1/2)
        size= max(r)   
        return size    
    
    def ico(self):
        self.atomtype=str(self.ui.Atom_Type_lineEdit.text())
        self.lattpar=float(self.ui.Lattice_Parameter_lineEdit.text())
        nshell=int(self.ui.Number_Shell_lineEdit.text())
        ico=Icosahedron(self.atomtype, nshell, self.lattpar)
        vertices_list=ico.get_positions()
        atom_number=len(vertices_list)
        path_to_file=self.path+'/Icosahedron_%s'%self.atomtype+'_%d'%nshell+'shells.xyz'
        xyz=open(path_to_file,'w')
        xyz.write('\t'+str(atom_number)+'\n \n')
        xs=list(zip(*vertices_list))[0]
        ys=list(zip(*vertices_list))[1]
        zs=list(zip(*vertices_list))[2]
        element=str(self.atomtype)
        for i in range(atom_number):
            xyz.write(str(element)+'\t'+str(xs[i])+'\t'+str(ys[i])+'\t'+str(zs[i])+'\n')
        xyz.close()
        
        size=2*self.ico_size(ico)
        
        self.ui.Console_textEdit.insertPlainText("XYZ file successfully generated in %s"%path_to_file+'\n')
        self.ui.Console_textEdit.insertPlainText("The cluster is contained in a sphere of diameter: %f"%size+"angströms. \n") 
        return 
    
    def deca(self):
        self.atomtype=str(self.ui.Atom_Type_lineEdit.text())
        self.lattpar=float(self.ui.Lattice_Parameter_lineEdit.text())
        p=int(self.ui.p_lineEdit.text())
        q=int(self.ui.q_lineEdit.text())
        r=int(self.ui.r_lineEdit.text())
        deca=Decahedron(self.atomtype,p,q,r, self.lattpar)
        vertices_list=deca.get_positions()
        atom_number=len(vertices_list)
        path_to_file=self.path+'/Decahedron_%s'%self.atomtype+'_p=%d'%p+'_q=%d'%q+'_r=%d'%r+'.xyz'
        xyz=open(path_to_file,'w')
        xyz.write('\t'+str(atom_number)+'\n \n')
        xs=list(zip(*vertices_list))[0]
        ys=list(zip(*vertices_list))[1]
        zs=list(zip(*vertices_list))[2]
        element=str(self.atomtype)
        for i in range(atom_number):
            xyz.write(str(element)+'\t'+str(xs[i])+'\t'+str(ys[i])+'\t'+str(zs[i])+'\n')
        xyz.close()
        size=2*self.ico_size(deca)
        
        self.ui.Console_textEdit.insertPlainText("XYZ file successfully generated in %s"%path_to_file+'\n')
        self.ui.Console_textEdit.insertPlainText("The cluster is contained in a sphere of diameter: %f"%size+"angströms. \n") 
        return
    
    def octa(self):
        self.atomtype=str(self.ui.Atom_Type_lineEdit.text())
        self.lattpar=float(self.ui.Lattice_Parameter_lineEdit.text())
        length=int(self.ui.length_lineEdit.text())
        cutoff=int(self.ui.cutoff_lineEdit.text())
        
        octa=Octahedron(self.atomtype,length,cutoff, self.lattpar)
        vertices_list=octa.get_positions()
        atom_number=len(vertices_list)
        path_to_file=self.path+'/Octahedron_%s'%self.atomtype+'_length=%d'%length+'_cutoff=%d'%cutoff+'.xyz'
        xyz=open(path_to_file,'w')
        xyz.write('\t'+str(atom_number)+'\n \n')
        xs=list(zip(*vertices_list))[0]
        ys=list(zip(*vertices_list))[1]
        zs=list(zip(*vertices_list))[2]
        element=str(self.atomtype)
        for i in range(atom_number):
            xyz.write(str(element)+'\t'+str(xs[i])+'\t'+str(ys[i])+'\t'+str(zs[i])+'\n')
        xyz.close()
        size=2*self.ico_size(octa)
        
        self.ui.Console_textEdit.insertPlainText("XYZ file successfully generated in %s"%path_to_file+'\n')
        self.ui.Console_textEdit.insertPlainText("The cluster is contained in a sphere of diameter: %f"%size+"angströms. \n") 
        return  
    
    def jmol(self):
        strufile=QtWidgets.QFileDialog.getOpenFileName( self,"Open xyz structure file","%s"%self.path,"Structure File *.xyz" )
        
        try:
            os.chdir(self.path)
            os.system("jmol %s"%strufile[0])
        except:
            self.ui.Console_textEdit.insertPlainText("Error: Jmol is not installed on your computer. You can alternatively view your structure with VESTA. \n")
            #self.ui.Console_textEdit.insertPlainText("Error: Jmol is not installed on your computer. You can alternatively view your structure with VESTA. \n")
        return
    ##### Cluster from cif
    def Make_XYZ_Sphere(self,cif_file,radius,xyz_filename):
        
        """
        produces a xyz file with the following structure
        Total N° of atoms
        Atom type and N)s os atoms of this type
        list atoms with x y z absolute coordinates
        """
        sphere=makeSphere(loadStructure(cif_file),radius)
        #Since diffpy.structure.expansion.makesphere computes fractional coordinates, 
        # each xyz coordinate must be multiplied by the corresponding lattice parameter.
        a=sphere.lattice.a
        b=sphere.lattice.b
        c=sphere.lattice.c
        
        xs=sphere.x*a
        ys=sphere.y*b
        zs=sphere.z*c
        r=(sphere.x**2+sphere.y**2+sphere.z**2)**(1/2)
        #self.ui.Console_textEdit.insertPlainText(r)
        rs=(xs**2+ys**2+zs**2)**(1/2)
        #self.ui.Console_textEdit.insertPlainText(rs)
        composition=sphere.composition
        #element=sphere.element
        atom_number=len(sphere)
        xyz=open(str(xyz_filename),'w')
        xyz.write(str(atom_number)+'\n')
        xyz.write(str(composition)+'\n')
        for i in range(atom_number):
            element=''.join(filter(str.isalpha, sphere.element[i]))
            xyz.write(element+'\t'+str(xs[i])+'\t'+str(ys[i])+'\t'+str(zs[i])+'\n')
        xyz.close()
        return
    
    def Make_XYZ_Ellipsoid(self,cif_file,a_radius,b_radius,c_radius,xyz_filename):
        
        """
        produces a xyz file with the following structure
        Total N° of atoms
        Atom type and N)s os atoms of this type
        list atoms with x y z absolute coordinates
          
        
        """
        ellipse=makeEllipsoid(loadStructure(cif_file),a_radius,b_radius,c_radius)
        #Since diffpy.structure.expansion.makesphere computes fractional coordinates, 
        # each xyz coordinate must be multiplied by the corresponding lattice parameter.
        a=ellipse.lattice.a
        b=ellipse.lattice.b
        c=ellipse.lattice.c
        xs=ellipse.x*a
        ys=ellipse.y*b
        zs=ellipse.z*c
        composition=ellipse.composition
        #element=ellipse.element
        atom_number=len(ellipse)
        xyz=open(str(xyz_filename),'w')
        xyz.write(str(atom_number)+'\n')
        xyz.write(str(composition)+'\n')
        for i in range(atom_number):
            element=''.join(filter(str.isalpha, ellipse.element[i]))
            xyz.write(element+'\t'+str(xs[i])+'\t'+str(ys[i])+'\t'+str(zs[i])+'\n')
        xyz.close()
        return
    
    def Make_XYZ_Disk(self,cif_file, axis, radius, thickness,xyz_filename):
        """
        Parameters
        ----------
        cif_file : str
            name of the cif file used 
        axis : str 
        rotation axis a, b or c
        
        radius : float
            radius of the disk/cylinder
        thickness : float
            thickness of the disk/cylinder
        xyz_filename : str
            Name of the xyz file
    
        Returns
        -------
        None.
    
        """
        import numpy as np
        from diffpy.Structure.expansion.supercell_mod import supercell
        from diffpy.structure.expansion.shapeutils import findCenter
        # three step process
        # (1) create a supercell block of sufficient size
        # (2) find atom nearest to the center of mass
        # (3) remove any atoms outside of the desired cylinder
        
        structure=loadStructure(cif_file)
        a=structure.lattice.a
        b=structure.lattice.b
        c=structure.lattice.c
        
        
        if axis=='b':
            # create supercell block of sufficient size
            MX=int(np.ceil(thickness / a + 1))
            MY=2*int(np.ceil(radius/b))
            MZ=2*int(np.ceil(radius/c))
            block=supercell(structure, [MX, MY, MZ])
            # (2)find atom nearest to the center of mass
            center_index=findCenter(block)
            #(2)' calculate the xyz coordinates of the center of mass in cartesian coordinates
            center_cartn = block.xyz_cartn.mean(axis=0)
            
            #(3) Remove atoms outside the desired cylinder
            # Calculation of distances
            vfromcenter = block.xyz_cartn - center_cartn 
            xoffset = vfromcenter[:, 0]
           
            #roffset=SQRT[(X2-X1)²+(Y2-Y1)²]
            roffset = np.power(vfromcenter[:,1:], 2).sum(axis=1)**0.5
            
            #Generate an array of booleans indicating wether the atom is outside the cylinder 
            #with 2 conditions on z and r
            isoutside = (np.abs(xoffset) > thickness / 2)
            isoutside = np.logical_or(isoutside, (roffset > radius))
            #remove atoms
            cylinder = block - block[isoutside]
            cylinder.write(xyz_filename, 'xyz')
        if axis=='a':
            MY=int(np.ceil(thickness / b + 1))
            MX=2*int(np.ceil(radius/a))
            MZ=2*int(np.ceil(radius/c))
            block=supercell(structure, [MX, MY, MZ])
            # (2)find atom nearest to the center of mass
            center_index=findCenter(block)
            #(2)' calculate the xyz coordinates of the center of mass in cartesian coordinates
            center_cartn = block.xyz_cartn.mean(axis=0)
            
            #(3) Remove atoms outside the desired cylinder
            # Calculation of distances
            vfromcenter = block.xyz_cartn - center_cartn 
            
            yoffset = vfromcenter[:, 1]
            #roffset=SQRT[(X2-X1)²+(Y2-Y1)²]
            roffset = np.power(vfromcenter[:,[0,2]], 2).sum(axis=1)**0.5
            #Generate an array of booleans indicating wether the atom is outside the cylinder 
            #with 2 conditions on z and r
            isoutside = (np.abs(yoffset) > thickness / 2)
            isoutside = np.logical_or(isoutside, (roffset > radius))
            #remove atoms
            cylinder = block - block[isoutside]
            cylinder.write(xyz_filename, 'xyz')
        if axis=='c':
            MZ=int(np.ceil(thickness / c + 1))
            MY=2*int(np.ceil(radius/b))
            MX=2*int(np.ceil(radius/a))
            block=supercell(structure, [MX, MY, MZ])
            # (2)find atom nearest to the center of mass
            center_index=findCenter(block)
            #(2)' calculate the xyz coordinates of the center of mass in cartesian coordinates
            center_cartn = block.xyz_cartn.mean(axis=0)
            
            #(3) Remove atoms outside the desired cylinder
            # Calculation of distances
            vfromcenter = block.xyz_cartn - center_cartn 
            #self.ui.Console_textEdit.insertPlainText (np.choose(vfromcenter,[0,1]))
            zoffset = vfromcenter[:, 2]
            #roffset=SQRT[(X2-X1)²+(Y2-Y1)²]
            #self.ui.Console_textEdit.insertPlainText (vfromcenter[:,1:])
            roffset = np.power(vfromcenter[:,:2], 2).sum(axis=1)**0.5
            #Generate an array of booleans indicating wether the atom is outside the cylinder 
            #with 2 conditions on z and r
            isoutside = (np.abs(zoffset) > thickness / 2)
            isoutside = np.logical_or(isoutside, (roffset > radius))
            #remove atoms
            cylinder = block - block[isoutside]
            cylinder.write(xyz_filename, 'xyz')
        return    
    def openciffile(self):
        self.cifpath= QtWidgets.QFileDialog.getOpenFileName( self,"Open cif file",self.path,'*.cif')
        self.ui.cifpath_lineEdit.setText(os.path.basename(self.cifpath[0]))
        
        self.ui.Console_textEdit.insertPlainText('Cif file is %s'%self.cifpath[0]+'.\n')
        return self.cifpath[0]
    def outputspec(self):
        self.outputpath=QtWidgets.QFileDialog().getExistingDirectory(self,'Select directory where to store the XYZ file')
        self.ui.Path_lineEdit.setText(str(self.outputpath))
        return self.outputpath
    def outputspec2(self):
        self.outputpath=QtWidgets.QFileDialog().getExistingDirectory(self,'Select directory where to store the XYZ file')
        self.ui.Path2_lineEdit.setText(str(self.outputpath))
        return self.outputpath    
     
    def makexyz(self):
        cif_file=self.cifpath[0]
        if self.ui.SpherecheckBox.isChecked()==True:
            radius=float(self.ui.radius_lineEdit.text())
            self.xyz_filename=str(self.outputpath)+'/Sphere_%s' %radius +'.xyz'
            self.Make_XYZ_Sphere(cif_file, radius, self.xyz_filename)
        
        if self.ui.EllipsoidcheckBox.isChecked()==True:
            a_radius=float(self.ui.a_radiuslineEdit.text())
            b_radius=float(self.ui.b_radiuslineEdit.text())
            c_radius=float(self.ui.c_radiuslineEdit.text())
            self.xyz_filename=str(self.outputpath)+'/Ellispoid_%s' %a_radius+'_%s'%b_radius+'%s'%c_radius+'.xyz'
            self.Make_XYZ_Ellipsoid(cif_file, a_radius, b_radius, c_radius, self.xyz_filename)
            
        if self.ui.checkBox.isChecked()==True:
            axis=str(self.ui.disk_axislineEdit.text())
            radius=float(self.ui.disk_radiuslineEdit.text())
            thickness=float(self.ui.disk_thicklineEdit.text())
            self.xyz_filename=str(self.outputpath)+'/Disk_axis_%s' %axis+'_radius_%s' %radius +'_thick_%s' %thickness +'.xyz'
            self.Make_XYZ_Disk(cif_file, axis, radius, thickness, self.xyz_filename)
        self.ui.Console_textEdit.insertPlainText('XYZ file succesfully generated \n')
        self.ui.Console_textEdit.insertPlainText ('Plot to check \n')
        return self.xyz_filename    
    
    
    
    ##################### PDF calc simulation tab #######################
    
    def open_simu_stru(self):
        simu_strufile=QtWidgets.QFileDialog.getOpenFileName( self,"Open xyz structure file","%s"%self.path,"Structure File (*.xyz *.cif)" )
        self.simu_strufile=simu_strufile[0]
        res=str(os.path.basename(self.simu_strufile))
        self.ui.simu_Stru_File_lineEdit.setText(res)
        return self.simu_strufile
    
    def iofq(self):
        from complex_form_factor import _f_complex,_f0,atomicformfactor_nist
        """Calculate I(Q) (X-ray) using the Debye Equation.
        
        I(Q) = 2 sum(i,j) f_i(Q) f_j(Q) sinc(rij Q) exp(-0.5 ssij Q**2)
        (The exponential term is the Debye-Waller factor.)
        
        S   --  A diffpy.structure.Structure instance. It is assumed that the
                structure is that of an isolated scatterer. Periodic boundary
                conditions are not applied.
        q   --  The q-points to calculate over.
        atom_list
        f_at_list: np.Arrays extracted from form factor calculations
        """
        qmin=float(self.ui.iq_qmin_lineEdit.text())
        qmax=float(self.ui.iq_qmax_lineEdit.text())
        qstep=float(self.ui.iq_qstep_lineEdit.text())
        wavelength=float(self.ui.wavelength_lineEdit.text())
        q=np.arange(qmin,qmax,qstep)
        # The functions we need
        sinc = np.sinc
        exp = np.exp
        pi = np.pi
        # read structure
        S = Structure()
        S.read(str(self.simu_strufile))
        S.Uisoequiv=float(self.ui.Uij_lineEdit.text())
        """
        h5filename='/home-local/ratel-ra/Documents/Ressources/fact_at_Ka_Mo_qstep_2e-3.h5'
        f=h5py.File(h5filename,'r') 
        nb_grp=0
        for group in f:
                nb_grp+=1            
        atom_list=np.zeros(nb_grp,dtype='U2')
        f_at_list=np.zeros((nb_grp,len(q)),dtype='complex128')
        k=0
        for group in f:
                atom=(str(group))
                atom_list[k]=atom
                f_at=np.array(f[group+'/fact_at']); f_at_list[k]=f_at
                k+=1 
        f.close()
        """
        # The brute-force calculation is very slow. Thus we optimize a little bit.
        # The precision of distance measurements
        deltad = 1e-6
        dmult = int(1/deltad)
        deltau = deltad**2
        umult = int(1/deltau)
        pairdict = {}
        elcount = {}
        n = len(S)
        for i in range(n):
                
        # count the number of each element
                eli = S[i].element
                #f_at_=np.zeros((nb_grp,len(q)),dtype='complex128')
                m = elcount.get(eli, 0)
                elcount[eli] = m + 1
        
                for j in range(i + 1, n):

                        elj = S[j].element
                
                        # Get the pair
                        els = [eli, elj]
                        els.sort()
                
                        # Get the distance to the desired precision
                        d = S.distance(i, j)
                        D = int(d*dmult)
                
                        # Get the DW factor to the same precision
                        ss = S[i].Uisoequiv + S[j].Uisoequiv
                        SS = int(ss*umult)
                
                        # Record the multiplicity of this pair
                        key = (els[0], els[1], D, SS)
                        mult = pairdict.get(key, 0)
                        pairdict[key] = mult + 1

    # Now we can calculate IofQ from the pair dictionary. Making the dictionary
    # first reduces the amount of calls to sinc and exp we have to make.

    # First we must cache the scattering factors
        fdict = {}
        for el in elcount:
                #print(el)
                try:
                    fdict[el]=_f_complex(el,qmin,qmax,qstep,wavelength)
                    self.ui.Console_textEdit.insertPlainText('X-Ray form factor of element %s'%str(el)+' computed in its complex form, including dispersive components \n')
                except:
                    fdict[el]=_f0(qmin,qmax,qstep,el)
                    self.ui.Console_textEdit.insertPlainText('X-Ray form factor of element %s'%str(el)+' computed as f0 \n')
                    #fdict[el] = f_at_list[list(atom_list).index(el)]
                #print('form factor for element %s'%str(el)+' is:',fdict[el])

    # Now we can compute I(Q) for the i != j pairs
        y = 0
        x = q * deltad / pi
        for key, mult in pairdict.items():
                eli = key[0]
                elj = key[1]
                fi = fdict[eli]
                fj = fdict[elj]
                D = key[2]
                SS = key[3]
                # Debye Waller factor
                DW = exp(-0.5 * SS * deltau * q**2)
                # Note that numpy's sinc(x) = sin(x*pi)/(x*pi)
                y += np.abs(fi * fj) * mult * sinc(x * D) * DW

    # We must multiply by 2 since we only counted j > i pairs.
        y *= 2
        
        # Now we must add in the i == j pairs.
        for el, f in fdict.items():
                y += np.abs(f**2) * elcount[el]

    

        return np.array([q,y])    
    
    
    def pdfcalc(self):
        
        structuretype=str(os.path.splitext(self.simu_strufile)[1])
        qmin=float(self.ui.qmin_lineEdit.text())
        qmax=float(self.ui.qmax_lineEdit.text())
        qdamp=float(self.ui.qdamp_lineEdit.text())
        qbroad=float(self.ui.qbroad_lineEdit.text())
        rmin=float(self.ui.rmin_lineEdit.text())
        rmax=float(self.ui.rmax_lineEdit.text())
        rstep=float(self.ui.rstep_lineEdit.text())
        config = {
            'qmin': qmin,
            'qmax': qmax,
            'qdamp':qdamp,
            'qbroad':qbroad,
            'rmin':rmin,
            'rmax': rmax,
            'rstep':rstep
        }
        pal = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(pal)
        
        gr_outputfilename_path=str(os.path.splitext(self.simu_strufile)[0]+'_calc_diffpy.gr')
        iq_outputfilename_path=str(os.path.splitext(self.simu_strufile)[0]+'_calc_diffpy.iq')
        gr_outputfilename=os.path.basename(gr_outputfilename_path)
        iq_outputfilename=os.path.basename(iq_outputfilename_path)
        self.outputfullpath_gr=str(self.path)+'/'+str(gr_outputfilename)
        self.outputfullpath_iq=str(self.path)+'/'+str(iq_outputfilename)
        
        if structuretype == '.xyz':
            structure=loadStructure(self.simu_strufile)
            
            calc_db = createParallelCalculator(
                DebyePDFCalculator(),
                pal,
                pool.imap_unordered
            )
            """
            calc_db = createParallelCalculator(
                PDFCalculator(),
                pal,
                pool.imap_unordered
            )
            """
            for atoms in structure:
                atoms.Uisoequiv=0.004
            # Total scattering
            r, g_tot = calc_db(structure, **config)
               
        if structuretype == '.cif':
            stru=self.simu_strufile
            parser=getParser('cif')
            structure=parser.parseFile(stru)
            # set Uiso for all atoms since no PDF is calculated when Uiso not provided (not always true in cif files!)
            for atoms in structure:
                atoms.Uisoequiv=0.004
            pdfcalc=PDFCalculator()
            calc = createParallelCalculator(
                pdfcalc,
                pal,
                pool.imap_unordered
            )

            # Total scattering
            r, g_tot = calc(structure, **config)
        
        # CalculateI(q) on the WAXS q range
        iofq=self.iofq()
        q=iofq[0]; iq=iofq[1]
        
                
        
        
        # plot and save
        X=np.stack([r,g_tot],axis=1)
        np.savetxt(self.outputfullpath_gr, X)
        X=np.stack([q,iq],axis=1)
        np.savetxt(self.outputfullpath_iq, X)            
        """
        plt.figure(1)
        plt.plot(r,g_tot,'-b') 
        plt.xlabel('r ($\mathrm{\AA}$)')
        plt.ylabel('G(r)')
        plt.show()
        """
        fig, (ax1,ax2)=plt.subplots(2,1)
        ax1.plot(q,iq/np.max(iq),'-g')
        ax1.set_xlabel('q  1/($\mathrm{\AA}$)')
        ax1.set_ylabel('Normalized I(q)')        
        
        ax2.plot(r,g_tot,'-b') 
        ax2.set_xlabel('r ($\mathrm{\AA}$)')
        ax2.set_ylabel('G(r)')
        plt.show()
        return
    
    def open_exp_pdf(self):
        exp_pdf=QtWidgets.QFileDialog.getOpenFileName( self,"Open experimental pdf file","%s"%self.path,"Pair Distribution Function file (*.gr)" )
        self.exp_pdf=exp_pdf[0]
        res=str(os.path.basename(self.exp_pdf))
        self.ui.ExpFile_TextEdit.setText(res)
        return self.exp_pdf
    
    def open_fit_strufile(self):
        fit_strufile=QtWidgets.QFileDialog.getOpenFileName( self,"Open xyz structure file","%s"%self.path,"Structure File (*.xyz *.cif)" )
        self.fit_strufile=fit_strufile[0]
        res=str(os.path.basename(self.fit_strufile))
        self.ui.CifFile_TextEdit.setText(res)
        return self.fit_strufile
        
    def make_recipe_cif(self,cif_path1, dat_path):
        Qdampflag=self.ui.fitQdamp_checkBox.isChecked()
        p_cif1 = getParser('cif')
        stru1 = p_cif1.parseFile(cif_path1)
        sg1 = p_cif1.spacegroup.short_name
        PDF_RMIN=float(self.ui.rmin_lineEdit.text())
        PDF_RMAX=float(self.ui.rmax_lineEdit.text())
        PDF_RSTEP=float(self.ui.rstep_lineEdit.text())
        QBROAD_I=float(self.ui.qbroad_lineEdit.text())
        QDAMP_I=float(self.ui.qdamp_lineEdit.text())
        QMIN=float(self.ui.qmin_lineEdit.text())
        QMAX=float(self.ui.qmax_lineEdit.text())
        profile = Profile()
        parser = PDFParser()
        parser.parseFile(dat_path)
        profile.loadParsedData(parser)
        profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

        generator_crystal1 = DebyePDFGenerator("G1")
        generator_crystal1.setStructure(stru1, periodic=True)

        contribution = FitContribution("crystal")
        contribution.addProfileGenerator(generator_crystal1)
        
        
        # If you have a multi-core computer (you probably do), run your refinement in parallel!
        if self.RUN_PARALLEL:
            try:
                import psutil
                import multiprocessing
                from multiprocessing import Pool
            except ImportError:
                self.ui.Console_textEdit.insertPlainText("\nYou don't appear to have the necessary packages for parallelization")
            syst_cores = multiprocessing.cpu_count()
            cpu_percent = psutil.cpu_percent()
            avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
            ncpu = int(np.max([1, avail_cores]))
            pool = Pool(processes=ncpu)
            generator_crystal1.parallel(ncpu=ncpu, mapfunc=pool.map)
            

        contribution.setProfile(profile, xname="r")

        
        if self.ui.sphericalCF_checkBox.isChecked():
            contribution.registerFunction(sphericalCF, name="fsphere")
            contribution.setEquation("s1*G1*fsphere")
            
        else:
            contribution.setEquation("s1*G1")
        # 15: Create the Fit Recipe object that holds all the details of the fit.
        recipe = FitRecipe()
        recipe.addContribution(contribution)

        # 16: Add, initialize, and tag the two scale variables.
        
        recipe.addVar(contribution.s1, float(self.ui.scaleFactor_lineEdit.text()), tag="scale")
        if self.ui.sphericalCF_checkBox.isChecked():
            recipe.addVar(contribution.psize,float(self.ui.diameter_lineEdit.text()),tag="radius")
            self.ui.Console_textEdit.insertPlainText("psize variable added to recipe")
       
        
        for name, generator, space_group in zip([""],
                                                [generator_crystal1],
                                                [sg1]):

            # 18b: Initialize the instrument parameters, Q_damp and Q_broad, and
            # assign Q_max and Q_min for each phase.
            generator.qdamp.value = QDAMP_I
            generator.qbroad.value = QBROAD_I
            generator.setQmax(QMAX)
            generator.setQmin(QMIN)

            # 18c: Get the symmetry equivalent parameters for each phase.
            spacegroupparams = constrainAsSpaceGroup(generator.phase,
                                                     space_group)
            # 18d: Loop over and constrain these parameters for each phase.
            # Each parameter name gets the loop index 'i' appeneded so there are not
            # parameter name collisions.
            for par in spacegroupparams.latpars:
                recipe.addVar(par,
                              name=f"{par.name}_{name}",
                              fixed=False,
                              tag='lat')
            for par in spacegroupparams.adppars:
                recipe.addVar(par,
                              name=f"{par.name}_{name}",
                              fixed=False,
                              tag='adp')
                recipe.restrain(f"{par.name}_{name}",lb=0,ub=0.5,scaled=True,sig=0.00001)
            
            recipe.addVar(generator.delta2, name=f"Delta2_{name}",
                         value=float(self.ui.delta2_lineEdit.text()), tag="d2")
            recipe.restrain(f"Delta2_{name}",lb=0,ub=10,scaled=True,sig=0.00001)
            recipe.restrain(f"Delta2_{name}",
                            lb=0.0,
                            ub=12.0,
                            scaled=True,
                            sig=0.00001)
        #19 bis define shape function for G(r) calculation using sphercialCF from diffpy
       
        recipe.crystal.registerFunction(sphericalCF,name='recipe')
        # 20: Return the Fit Recipe object to be optimized.
        if Qdampflag:
            recipe.addVar(generator_crystal1.qdamp,
                          fixed=False,
                          name="Calib_Qdamp",
                          value=QDAMP_I,
                          tag="inst")
        
            recipe.addVar(generator_crystal1.qbroad,
                          fixed=False,
                          name="Calib_Qbroad",
                          value=QBROAD_I,
                          tag="inst")            
        return recipe

        
    def make_recipe_xyz(self,xyz_path, dat_path):
        PDF_RMIN=float(self.ui.rmin_lineEdit.text())
        PDF_RMAX=float(self.ui.rmax_lineEdit.text())
        PDF_RSTEP=float(self.ui.rstep_lineEdit.text())
        QBROAD_I=float(self.ui.qbroad_lineEdit.text())
        QDAMP_I=float(self.ui.qdamp_lineEdit.text())
        QMIN=float(self.ui.qmin_lineEdit.text())
        QMAX=float(self.ui.qmax_lineEdit.text())
        ZOOMSCALE_I=1
        UISO_I=0.005
        stru1 = Structure(filename=str(xyz_path))

        profile = Profile()
        parser = PDFParser()
        parser.parseFile(dat_path)
        profile.loadParsedData(parser)
        profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

        # 10: Create a Debye PDF Generator object for the discrete structure model.
        generator_cluster1 = DebyePDFGenerator("G1")
        generator_cluster1.setStructure(stru1, periodic=False)

        # 11: Create a Fit Contribution object.
        contribution = FitContribution("cluster")
        contribution.addProfileGenerator(generator_cluster1)
                
        # If you have a multi-core computer (you probably do), run your refinement in parallel!
        if self.RUN_PARALLEL:
            try:
                import psutil
                import multiprocessing
                from multiprocessing import Pool
            except ImportError:
                self.ui.Console_textEdit.insertPlainText("\nYou don't appear to have the necessary packages for parallelization")
            syst_cores = multiprocessing.cpu_count()
            cpu_percent = psutil.cpu_percent()
            avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
            ncpu = int(np.max([1, avail_cores]))
            pool = Pool(processes=ncpu)
            generator_cluster1.parallel(ncpu=ncpu, mapfunc=pool.map)
            
        contribution.setProfile(profile, xname="r")

        # 13: Set an equation, based on your PDF generators. 
        contribution.setEquation("s1*G1")

        # 14: Create the Fit Recipe object that holds all the details of the fit.
        recipe = FitRecipe()
        recipe.addContribution(contribution)

        # 15: Initialize the instrument parameters, Q_damp and Q_broad, and
        # assign Q_max and Q_min.
        generator_cluster1.qdamp.value = QDAMP_I
        generator_cluster1.qbroad.value = QBROAD_I
        generator_cluster1.setQmax(QMAX)
        generator_cluster1.setQmin(QMIN)

        # 16: Add, initialize, and tag variables in the Fit Recipe object.
        # In this case we also add psize, which is the NP size.
        recipe.addVar(contribution.s1, float(self.ui.scaleFactor_lineEdit.text()), tag="scale")

        # 17: Define a phase and lattice from the Debye PDF Generator
        # object and assign an isotropic lattice expansion factor tagged
        # "zoomscale" to the structure. 

        phase_cluster1 = generator_cluster1.phase

        lattice1 = phase_cluster1.getLattice()

        recipe.newVar("zoomscale", ZOOMSCALE_I, tag="lat")

        recipe.constrain(lattice1.a, 'zoomscale')
        recipe.constrain(lattice1.b, 'zoomscale')
        recipe.constrain(lattice1.c, 'zoomscale')

        # 18: Initialize an atoms object and constrain the isotropic
        # Atomic Displacement Paramaters (ADPs) per element. 

        atoms1 = phase_cluster1.getScatterers()
        recipe.newVar("Uiso", UISO_I, tag="adp")
        for atom in atoms1:
            recipe.constrain(atom.Uiso, "Uiso")
            recipe.restrain("Uiso",lb=0,ub=1,scaled=True,sig=0.00001)
        recipe.addVar(generator_cluster1.delta2, name="delta2", value=float(self.ui.delta2_lineEdit.text()), tag="d2")
        recipe.restrain("delta2",lb=0,ub=12,scaled=True,sig=0.00001)
        return recipe
    
        
        
        
    # 21 We create a useful function 'plot_results' for writing a plot of the fit to disk.
    def plot_results(self,recipe, fig_name):
        """
        Creates plots of the fitted PDF and residual, and writes them to disk
        as *.pdf files.

        Parameters
        ----------
        recipe :    The optimized Fit Recipe object containing the PDF data
                    we wish to plot.
        fig_name :  bPath object, the full path to the figure file to create..

        Returns
        ----------
        None
        """
        if not isinstance(fig_name, Path):
            fig_name = Path(fig_name)

        plt.clf()
        plt.close('all')
        structuretype=str(os.path.splitext(self.fit_strufile)[1])

        if structuretype=='.cif':
            r = recipe.crystal.profile.x
            g = recipe.crystal.profile.y
            gcalc = recipe.crystal.profile.ycalc
       
        if structuretype=='.xyz':
            r = recipe.cluster.profile.x
            g = recipe.cluster.profile.y
            gcalc = recipe.cluster.profile.ycalc

        # Make an array of identical shape as g which is offset from g.
        diff = g - gcalc
        diffzero = (min(g)-np.abs(max(diff))) * \
            np.ones_like(g)

        # Calculate the residual (difference) array and offset it vertically.
        diff = g - gcalc + diffzero

        # Get the Ni and Si scale terms
        #ni_scale = recipe.s2.value*(1.0-recipe.s1_Co.value)
        #co_scale = recipe.s2.value*recipe.s1_Co.value

        

        # Change some style detials of the plot
        mpl.rcParams.update(mpl.rcParamsDefault)
        
        # Create a figure and an axis on which to plot
        fig, ax1 = plt.subplots(1, 1)

        # Plot the difference offset line
        ax1.plot(r, diffzero, lw=1.0, ls="--", c="black")

        # Plot the measured data
        ax1.plot(r,
                 g,
                 ls="None",
                 marker="o",
                 ms=5,
                 mew=0.2,
                 mfc="None",
                 label="G(r) Data")
        ax1.plot(r, diff, lw=1.2, label="G(r) diff")
        ax1.plot(r,gcalc,'g',label='G(r) calc')
        ax1.set_xlabel(r"r ($\mathrm{\AA}$)")
        ax1.set_ylabel(r"G ($\mathrm{\AA}$$^{-2}$)")
        ax1.tick_params(axis="both",
                        which="major",
                        top=True,
                        right=True)

        ax1.set_xlim(r[0], r[-1])
        ax1.legend(ncol=2)
        plt.tight_layout()
        plt.show()
        fig.savefig(fig_name.parent / f"{fig_name.name}.pdf", format="pdf")
        return
    
    def rundiffpy(self):
        from pathlib import Path
        self.ui.Console_textEdit.insertPlainText('Refinement in progress. Please wait. \n')
        
        PWD=Path(self.path)
        # Make some folders to store our output files.
        resdir = PWD / "res"
        fitdir = PWD / "fit"
        figdir = PWD / "fig"

        folders = [resdir, fitdir, figdir]

        for folder in folders:
            if not folder.exists():
                folder.mkdir()

        # Establish the location of the data and a name for our fit.
        gr_path = str(self.exp_pdf)
        FIT_ID=str(self.ui.fitID_lineEdit.text())
        basename = FIT_ID
       
        

        # Establish the full path of the CIF file with the structure of interest.
        stru_path = self.fit_strufile
        structuretype=str(os.path.splitext(stru_path)[1])
        # 23: Call 'make_recipe' to create our fit recipe.
        if structuretype=='.xyz':
            recipe = self.make_recipe_xyz(str(stru_path),str(gr_path))
        elif structuretype=='.cif':
            recipe = self.make_recipe_cif(str(stru_path),str(gr_path))
            

        # Tell the Fit Recipe we want to write the maximum amount of
        # information to the terminal during fitting.
        recipe.fithooks[0].verbose = 3

        recipe.fix("all")
        if self.ui.sphericalCF_checkBox.isChecked():
            tags= ["lat", "scale","radius", "adp", "d2", "all"]
            #tags= ["scale", "radius", "lat", "d2", "all"]
        else:
            if self.ui.fitQdamp_checkBox.isChecked():
                tags = ["lat", "scale", "adp", "d2","inst", "all"]
            else:
                tags = ["lat", "scale", "adp", "d2", "all"]
        #tags = ["lat", "scale", "d2", "all"]
        for tag in tags:
            recipe.free(tag)
            
            least_squares(recipe.residual, recipe.values, x_scale="jac")

        # Write the fitted data to a file.
        if structuretype=='.xyz':
            profile = recipe.cluster.profile
        elif structuretype=='.cif':
            profile = recipe.crystal.profile
        profile.savetxt(fitdir / f"{basename}.fit")

        # self.ui.Console_textEdit.insertPlainText the fit results to the terminal.
        res = FitResults(recipe)
        res.printResults()
        #self.ui.Console_textEdit.insertPlainText(res.printResults)

        # Write the fit results to a file.
        header = "%s"%str(basename)+".\n"
        res.saveResults(resdir / f"{basename}.res", header=header)

        # Write a plot of the fit to a (pdf) file.
        figname= figdir / basename
        
        self.plot_results(recipe, figname)

        # End of function
        return
    
  
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
