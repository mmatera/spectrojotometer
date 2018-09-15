
# coding: utf-8


from __future__ import print_function
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D




def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)



########################################################################33


def box_ellipse(A,r):
    """
    Determines the minimum edge half-length of the rectangular box 
    containing the ellipsoid defined by  x^t * A^t * A * x - r**2=0, with faces 
    parallel to the coordinate axes.
    """
    A = np.array(A)
    A = A.transpose().dot(A)
    size=len(A)
    widths=[]
    for i in range(size):
        setperp=[k for k in range(i)] + [k for k in range(i+1,size)]
        v = A[i,setperp]
        A22=A[setperp][:,setperp]
        print("setperp",setperp)
        Aperpinv=np.linalg.inv(A22)
        gamma = Aperpinv.dot(v)
        gamma = gamma.dot(v)
        widths.append(r/np.sqrt(A[i,i]-gamma))
    return widths




class MagneticModel:
    def __init__(self,atomic_pos, bravais_lat, 
                 bond_lists=None,
                 bond_names=None,
                 ranges=None,
                 supercell_size=2,
                 discretization=0.02,
                 magnetic_species=None,
                 onfly=True, model_label="default"):
        self.model_label = model_label
        self.cell_size = len(atomic_pos)
        self.coord_atomos = atomic_pos
        self.bravais_vectors = bravais_lat
        self.onfly = onfly

        if magnetic_species is None:
            self.magnetic_species = ["X" for i in atomic_pos]
        else:
            self.magnetic_species = []
            self.magnetic_species[:] = magnetic_species
        if len(bravais_lat) == 1:
            self.supercell = np.array([[site[idx] + i * self.bravais_vectors[0][idx]
                                       for idx in range(3)]
                            for i in range(-supercell_size,supercell_size+1)
                            for site in self.coord_atomos ])

        elif len(bravais_lat)== 2:
            self.supercell = np.array([[site[idx] + i * self.bravais_vectors[0][idx]
                                        + j * self.bravais_vectors[1][idx] 
                                       for idx in range(3)]
                            for i in range(-supercell_size,supercell_size+1)
                            for j in range(-supercell_size,supercell_size+1)
                            for site in self.coord_atomos ])
        elif len(bravais_lat) == 3:
            self.supercell = np.array([[site[idx] + i * self.bravais_vectors[0][idx]
                                        + j * self.bravais_vectors[1][idx] 
                                        + k * self.bravais_vectors[2][idx] 
                                       for idx in range(3)]
                            for i in range(-supercell_size,supercell_size+1)
                            for j in range(-supercell_size,supercell_size+1)
                            for k in range(-supercell_size,supercell_size+1)
                            for site in self.coord_atomos ])
        else:
            self.supercell = self.coord_atomos
        
        if ranges is None:
            maxdist = 0 
            for p in self.coord_atomos:
                for q in self.coord_atomos:
                    d = np.linalg.norm(p-q)
                    if d > maxdist:
                        maxdist = d
            ranges = [[0,d]]
        elif type(ranges) is float:
            ranges = [[0,ranges]]
        elif type(ranges) is list and type(ranges[0]) is float:
            ranges = [ranges]

        if bond_lists is not None:            
            self.bond_lists = bond_lists
            bond_distances = [0 for i in bond_lists]
            if bond_names is not None:
                self.bond_names = bond_names
            else:
                self.bond_names =[ "J"+str(i) for i in range(len(self.bond_lists))]
        else:
            self.bond_names = []
            if bond_names is not None:
                self.bond_names[:] = bond_names            
            self.bond_lists =[]
            self.bond_distances = []
            self.discretization = discretization
            self.generate_bonds(discretization,ranges)
            

            
    def remover_bond(self,idx):
        """
        Elimina el idx-esimo bond
        """
        self.bond_distances.pop(idx)
        self.bond_names.pop(idx)
        self.bond_lists.pop(idx)
        
    def remover_bond_by_name(self,name):
        """
        Elimina el bond de nombre <name>
        """
        try:
            idx = self.bond_names.index(name)
        except ValueError:
            eprint("bond "+ name + " not found")
            return
        
        self.remover_bond(idx)
        
        
    def generate_bonds(self, discretization, ranges, bond_names=None):
        """
        En base a la celda unidad, la red de Bravais y el tamaño de la 
        "super-celda" (cuantas copias de la celda unidad se consideran 
        como entorno) calcula los bonds,
        esto es, los pares de sitios magnéticos que interactúan según cada 
        una de las constantes de acoplamiento a determinar.
        """
        cell_size = self.cell_size 
        coord_atomos = self.coord_atomos
        bravais_vectors = self.bravais_vectors
        supercell = self.supercell
        old_bond_lists = self.bond_lists 
        old_bond_distances = self.bond_distances 
        old_bond_names = self.bond_names
        
        bond_lists = []
        bond_distances = []
        bond_type = []
        bond_names = []
        atom_type = self.magnetic_species
        
        def is_in_range(val):
            for r in ranges:
                if r[0]<val<r[1]:
                    return True
            return False
        
        for d, bt in sorted([(np.linalg.norm(q-p), set([atom_type[i],atom_type[j]]))
                             for i,q in enumerate(self.coord_atomos) 
                                             for j,p in enumerate(self.coord_atomos)]):
            dr = round(d/discretization)*discretization
            if not is_in_range(dr): 
                continue
            if dr!=0 and (dr not in bond_distances or bt not in bond_type ):
                bond_distances.append(dr)
                bond_lists.append([])
                bond_type.append(bt)
                    
        #supercell=np.array([p for p in supercell if p[2]>0])
        for p,x in enumerate(coord_atomos):        
            for q,y in enumerate(self.supercell):
                qred = q%cell_size            
                if(p<qred):                
                    d = x-y
                    d = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
                    if not is_in_range(d):
                        continue
                    bt = set([atom_type[p],atom_type[qred]])
                    for i in range(len(bond_distances)):
                        if np.abs(d-bond_distances[i])<discretization and bt == bond_type[i]:
                            bond_lists[i].append((p,qred))
                            
        self.bond_lists =  old_bond_lists +  bond_lists
        self.bond_distances =  old_bond_distances +  bond_distances

        
        nnames = len(self.bond_names)
        while len(self.bond_names)<len(self.bond_lists):
            while "J"+str(nnames+1) in self.bond_names:
                nnames = nnames + 1
            self.bond_names.append("J"+str(nnames+1))            
        return
            
        
    def print_equations(self,cm,ensname=None):
        print("\n\n Equations: \n============\n\n")
        jsname = []
        jsname[:] = self.bond_names
        jsname.append("E_0")
        if ensname is None:
            ensname = [ "E_"+str(i+1) for i in range(len(cm))]
        for i,row in enumerate(cm):
            eq=""
            for k,c in enumerate(row):
                cr = round(c*100)/100.
                if c>0:
                    if eq != "":
                        eq = eq + " + "
                    else:
                        eq = "  "
                    if cr != 1:
                        eq = eq +  str(cr)  +  " *"
                    eq = eq + " " + jsname[k]
                elif c<0:
                    if eq != "":
                        eq = eq + " "
                    eq = eq + "- "
                    if cr != -1:
                        eq = eq +  str(-cr)  +  " *"
                    eq = eq + " " + jsname[k]
            print(eq + " = " + ensname[i]+"\n")
        return


    def coefficient_matrix(self,configs,normalizar=True):
        """
        Devuelve la matriz que define el sistema de ecuaciones que vincula a las constantes
        de acoplamiento con las energías correspondientes a las configuraciones dadas.
        """
        rawcm = [np.array([
                         -sum([(-1)**sc[b[0]] * (-1)**sc[b[1]] for b in bondfamily ]) 
                         for sc in configs]) 
                        for bondfamily in self.bond_lists]
        if normalizar:
            cm = [ v-np.average(v) for v in rawcm]
        else:
            cm = rawcm
        cm = np.array(cm + 
                      [np.array([1. for sc in configs]) # Energía no magnética
                      ]).transpose()
        return cm

    def inv_min_sv_from_config(self,confs):
        """
        Dada una lista de configuraciones de espin, construye las ecuaciones y calcula la inversa del mínimo valor singular.
        Las ecuaciones son de la forma
        \Sum_{(ij)} J_{ij} S_i[c] S_j[c] = E_c)
        Aprovechamos que en la construcción de las ecuaciones garantizamos que las ecuaciones para
        la E0 y los js queden desacopladas (al restar en cada coeficiente el promedio sobre las configuraciones)
        """
        eqarray = self.coefficient_matrix(confs)
        eqarray = eqarray[:,0:-1]
        singularvalues = np.linalg.svd(eqarray)[1]
        if min(singularvalues)== 0:
            return 1e80
        cond_number = 1./min(singularvalues)
        return cond_number
    

    def compute_couplings(self, confs, energs, err_energs=.01,printeqs=False):
        """
        compute_couplings
        Given a set of configurations, and the energies calculated from 
        the ab-initio tools, estimates the values of the coupling 
        constants from the proposed model.
        
        confs: list of magnetic configurations
        energs: energies evaluated for each magnetic configuration
        err_energs: estimation of the maximum convergence error in energies
        printeqs: print the corresponding equations.
        """

        if printeqs:
            coeffs = self.coefficient_matrix(confs,normalizar=False)
            print("\nConfigurations:\n===============\n\n")
            for c in confs:
                print(c)
            self.print_equations(coeffs)

        coeffs = self.coefficient_matrix(confs,normalizar=True)

        # The choice in the way the equation is written allows to decouple
        # the determination of the coupling constants from the base
        # energy. This implies that the condition number associated to the
        # coupling constants should be evaluated from the reduced set of 
        # equations.
        rcoeffs = coeffs[:,0:-1]
        singularvalues = np.linalg.svd(rcoeffs)[1]
        cond_number = 1./min(singularvalues)
        if printeqs:
            print("\nInverse of the minimum singular value: ", cond_number,"\n\n")

        # js = (A^t A )^{-1} A^t En, i.e. least squares solution
        resolvent =  np.linalg.inv(coeffs.transpose().dot(coeffs)).dot(coeffs.transpose())
        js = resolvent.dot(energs)
#        deltaJ = [max(abs(js[:-1]))* err_energs/(max(abs(energs-js[-1]))) * cond_number for k in js]
        deltaJ = box_ellipse(2.*np.sqrt(len(coeffs))*coeffs,err_energs)
        model_chi = abs(coeffs.dot(js)-energs)/err_energs
        return (js, deltaJ, model_chi)

    def show_config(self, config, sp):
        """
        Esta función dibuja las redes con las correspondientes configuraciones de 
        espines.

           config es una lista con los estados de los átomos
           sp es un subplot donde hacer el gráfico
           si showbonds = True, dibuja lineas entre los sitios interactuantes
        """
        coord_atomos = self.coord_atomos
        fig = plt.figure()
        colors = ['r' if config[i] >0 else 'b' for i in range(self.cell_size)]
        ax = fig.add_subplot(sp, projection='3d')
        idx=0
        for i in range(len(coord_atomos)):
            p = coord_atomos[i]
            ax.text(p[0], p[1], p[2], s=i+1)
        ax.scatter(coord_atomos[:,0],coord_atomos[:,1],coord_atomos[:,2],c=colors)
        ax.set_ylim((-1,1))
        ax.set_xlim((-1,1))       
        return (fig,ax)

    def show_superlattice_bonds(self, nd=1.,bt=None,discretization=.02):
        """
        Muestra los sitios que interactúan entre sí a una cierta distancia.
        """
        coord_atomos = self.coord_atomos
        supercell = self.supercell
        fig = plt.figure()
        ax = fig.add_subplot(111)


        if bt is not None:
            nd=bond_distances[bt]
        colors=["r" for p in supercell]
        ax.scatter(supercell[:,0],supercell[:,1],c=colors)
        colors=["b" for p in coord_atomos]
        ax.scatter(coord_atomos[:,0],coord_atomos[:,1],c=colors)
        ax.set_ylim((-3,3))

        for p,x in enumerate(coord_atomos):
            for q,y in enumerate(supercell):
                qred = q%cell_size            
                if(p<qred):                
                    d = x-y
                    d = np.sqrt(d[0]**2+d[1]**2+ d[2]**2)
                    #print([(p,q),[np.abs(d-z) for z in [d0,d1p,d1x,d2p,d2x,d3p,d3x]]])
                    if np.abs(d-nd)<.01:
                        ax.add_artist(Line2D([x[0],y[0]],[x[1],y[1]]))
                elif(p>qred):
                    d = x-y
                    d = np.sqrt(d[0]**2+d[1]**2+ d[2]**2)
                    if d > 4.5:
                        continue
                    if np.abs(d-nd)<discretization:
                        ax.add_artist(Line2D([x[0],y[0]],[x[1],y[1]],linestyle=":"))
                elif(p==qred):
                    d = x-y
                    d = np.sqrt(d[0]**2+d[1]**2+ d[2]**2)
                    if np.abs(d-nd)<discretization:
                        ax.add_artist(Line2D([x[0],y[0]],[x[1],y[1]],linestyle="-."))
        return 


    def check_superlattice_bonds(self, nd=1.,bt=None,discretization=.02):
        """
        Muestra los sitios que interactúan entre sí a una cierta distancia.
        """
        coord_atomos = self.coord_atomos
        supercell = self.supercell
        if bt is not None:
            nd = self.bond_distances[bt]
        else:
            for k,di in self.bond_distances:
                if np.abs(d-nd)<discretization:
                    bt=k
                    break
        if bt is None:
            return 

        countbond=0
        for p,x in enumerate(coord_atomos):
            for q,y in enumerate(supercell):
                qred = q%cell_size            
                if(p<qred):                
                    d = x-y
                    d = np.sqrt(d[0]**2+d[1]**2+ d[2]**2)
                    if np.abs(d-nd)<discretization:
                        countbond = countbond+1
                        if (p,qred) not in self.bond_lists[bt]:
                            eprint ((p,qred), "missing")
                elif(p>qred):
                    d = x-y
                    d = np.sqrt(d[0]**2+d[1]**2+ d[2]**2)
                    if 9.9 > d > .7:
                        continue
                    if np.abs(d-nd)<discretization:
                        countbond = countbond + 1
                        if (qred,p) not in self.bond_lists[bt]:
                            eprint ((qred,p), "missing")
        if 2*len(bond_lists[bt]) != countbond:
            eprint (2*len(bond_lists[bt])," != ", countbond)
        return 

    def generate_configurations_onfly(self):
        size = self.cell_size
        for c in range(2**((size-1))):
            yield [c >> i & 1 for i in range(size-1,-1,-1)]

    def generate_random_configurations(self,t=10):
        size = self.cell_size
        for c in np.random.random_integers(0,2**((size-1)),t):
            yield [c >> i & 1 for i in range(size-1,-1,-1)]
    
    
    def find_optimal_configurations(self, start=None, num_new_confs=None, known=None, its=100, update_size=1):        
        if known is None:
            known = []
        
        if num_new_confs is None or num_new_confs + len(known) < len(self.bond_lists) + 1:
            num_new_confs = max(len(self.bond_lists)-len(known) +1 ,1)
        if start is None:
            repres = self.generate_random_configurations(2*num_new_confs)
            last_better = [q for q in repres]                                    
        else:            
            last_better = start

        num_confs = num_new_confs + len(known)
        inequiv_confs = last_better
        last_better_cn = self.inv_min_sv_from_config(last_better)
        cn = last_better_cn
        
        
        for it in range(its):

            #Dadas las configuraciones equivalentes, busca un subconjunto que optimize la dependencia 
            # de la energía con los parámetros

            #inequiv_confs=[ocho_a_diezyseis(c) for c in repres]

            #print("generando matriz de coeficientes para las configuraciones inequivalentes")
            #inequiv_confs=repres

            # Aquí se calculan los coeficientes del sistema de ecuaciones asociado al conjunto completo de configuraciones 
            # no equivalentes

            repres = self.generate_random_configurations(update_size)
            inequiv_confs= known + [q for q in repres] + inequiv_confs            
            coefs = self.coefficient_matrix(inequiv_confs)
            
            #Este es el algoritmo que uso para buscar las óptimas:
            # Descompongo coefs como SVD, y me quedo sólo con los vectores asociados a los valores singulares no nulos.
            # Luego, busco cuales configuraciones definen el soporte efectivo de los vectores singulares.
            v=((np.linalg.svd(coefs)[0])[0:len(coefs[0])]).transpose()
            v = v[len(known):]
            # Busco definir un soporte efectivo sobre los vectores singulares (que dan 
            # los índices en la lista de configuraciones).
            threshold=sorted([np.linalg.norm(z) for z in v])[-num_new_confs]

            # relevant guarda las configuraciones que lucen relevantes (porque tienen mayor peso en los 
            # vectores singulares)
            relevant=[]
            for j,val in enumerate(v):
                if np.linalg.norm(val)>=threshold:
                    if j not in relevant:
                        relevant.append(j)

            # Ordeno los índices de configuraciones relevantes y elimino duplicados.
            relevant = sorted(relevant)[:num_new_confs]
            relevant =  [k + len(known) for k in relevant]
            inequiv_confs = [inequiv_confs[k] for k in relevant]
            cn = self.inv_min_sv_from_config( known + inequiv_confs)
            # a partir de los índices, fabrico una lista más corta de configuraciones relevantes
            #print("Número de condición para el conjunto reducido de ",len(inequiv_confs)," elementos:",cn)
            if cn < last_better_cn:
                last_better_cn = cn
                last_better = inequiv_confs 
                eprint("it", it,"nuevo cn=", cn)
                eprint(inequiv_confs)
            else:
                inequiv_confs = last_better

        cn = last_better_cn
        inequiv_confs = last_better        

        return(last_better_cn, last_better)



    def save_cif(self, filename, bond_names=None):
        bravais_vectors = self.bravais_vectors
        
        with open(filename,"w") as fileout:
            head = """
#======================================================================
            
# CRYSTAL DATA

#----------------------------------------------------------------------

data_magnetic_model_1
            
_chemical_name_common                  """ + self.model_label + "\n"
            fileout.write(head)
            bbn = ["a","b","c"]
            bbnang = ["alpha","beta","gamma"]
            
            if len(bravais_vectors) == 1:
                fileout.write("_cell_length_a \t\t\t" + str(np.linalg.norm(bravais_vectors[0]))+"\n\n\n")
                fileout.write("loop_\n _space_group_symop_operation_xyz\t\t\t\n\'z\'\n")
            elif len(bravais_vectors) == 2:
                a = np.linalg.norm(bravais_vectors[0])
                b = np.linalg.norm(bravais_vectors[1])
                gamma = round(180/3.1415926 * bravais_vectors[0].dot(bravais_vectors[1])/(a*b))
                fileout.write("_cell_length_a \t\t\t" + str(a)+"\n")
                fileout.write("_cell_length_b \t\t\t" + str(b)+"\n")
                fileout.write("_cell_length_gamma \t\t\t" + str(gamma)+"\n\n")
                fileout.write("loop_\n _space_group_symop_operation_xyz\n\'x, y\'\n\n")
                
            elif len(bravais_vectors) == 3:
                a = np.linalg.norm(bravais_vectors[0])
                b = np.linalg.norm(bravais_vectors[1])
                c = np.linalg.norm(bravais_vectors[2])
                gamma =round(180/3.1415926 * np.arccos(bravais_vectors[0].dot(bravais_vectors[1])/(a*b)))
                alpha =round(180/3.1415926 * np.arccos(bravais_vectors[0].dot(bravais_vectors[2])/(a*c)))
                beta = round(180/3.1415926 * np.arccos(bravais_vectors[0].dot(bravais_vectors[2])/(c*b)))
                fileout.write("_cell_length_a \t\t\t" + str(a)+"\n")
                fileout.write("_cell_length_b \t\t\t" + str(b)+"\n")
                fileout.write("_cell_length_c \t\t\t" + str(c)+"\n")
                fileout.write("_cell_angle_alpha \t\t\t" + str(alpha)+"\n")
                fileout.write("_cell_angle_beta \t\t\t" + str(beta)+"\n")
                fileout.write("_cell_angle_gamma \t\t\t" + str(gamma)+"\n\n")
                fileout.write("loop_\n _space_group_symop_operation_xyz\t\t\t\n\'x, y, z\'\n\n")
            
            fileout.write("# Atom positions \n\n")
            fileout.write("loop_\n   _atom_site_label\n" + \
                          "_atom_site_occupancy\n" +  \
                          "_atom_site_fract_x\n" +  \
                          "_atom_site_fract_y\n" + \
                          "_atom_site_fract_z\n" + \
                          "_atom_site_adp_type\n" + \
                          "_atom_site_B_iso_or_equiv\n" + \
                          "_atom_site_type_symbol\n")

            bravaiscoords = self.coord_atomos.dot(np.linalg.inv( 
                                                    np.array(bravais_vectors)))
                                      
            for i,pos in enumerate(bravaiscoords):
                fileout.write(self.magnetic_species[i] + str(i+1) + "\t 1.\t" + \
                               str(round(1E5*pos[0])*1E-5) + "\t" +  \
                               str(round(1E5*pos[1])*1E-5) + "\t" + \
                               str(round(1E5*pos[2])*1E-5) + "\t" + \
                               "Biso \t" + \
                               "1 \t" + \
                               self.magnetic_species[i] + " \n")
            fileout.write("   \n")
            
            if len(self.bond_lists)>0 :
                fileout.write("# Bonds  \n")
                fileout.write("loop_\n")
                fileout.write("_geom_bond_atom_site_label_1\n")
                fileout.write("_geom_bond_atom_site_label_2\n")
                fileout.write("_geom_bond_distance\n")
                fileout.write("_geom_bond_label\n")                                                
                for k,bl in enumerate(self.bond_names):
                    for a,b in self.bond_lists[k]:
                        fileout.write(self.magnetic_species[a]+str(a+1) +"\t" +\
                                      self.magnetic_species[b]+str(b+1) +"\t" +\
                                      str(self.bond_distances[k]) +"\t" +\
                                      str(bl) +"\n")
                fileout.write("   \n")
        return True
            
            
            
            
   
def magnetic_model_from_file(filename, magnetic_atoms=["Mn","Fe","Co","Ni","Dy","Tb","Eu"],bond_names=None):
    if filename[-4:]==".cif" or filename[-4:]==".CIF":
        return magnetic_model_from_cif(filename, magnetic_atoms,bond_names)
    if filename[-7:]==".struct" or filename[-7:]==".STRUCT":
        return magnetic_model_from_wk2_struct(filename, magnetic_atoms,
                                              bond_names)    
    print("unknown file format")
    return -1

def magnetic_model_from_cif(filename, magnetic_atoms=["Mn","Fe","Co","Ni","Dy","Tb","Eu"],bond_names=None):    
    bravais_params = {}
    magnetic_positions = None
    bravais_vectors = None
    labels = None
    entries = None
    magnetic_species = []
    bond_labels = None
    bondlists = None
    bond_distances = []
    with open(filename,"r") as src:    
        for line in src:
            listrip = line.strip()
            if listrip[:13] =='_cell_length_':                                
                varvals = listrip[13:].split()
                bravais_params[varvals[0]] = float(varvals[1])
            elif listrip[:12] =='_cell_angle_':                
                varvals = line[12:].strip().split()
                bravais_params[varvals[0]] = float(varvals[1])*3.1415926/180.
            elif listrip[:5]=='loop_':                
                labels = []
                entries = []
                l = src.readline()
                listrip = l.strip()                 
                if listrip == '':                    
                    break
                if listrip!="" and line[0] == "#":                    
                    continue
                while listrip[0] == '_' or listrip[0] == '#':
                    if listrip != "" and line[0] == "#":                        
                        continue
                    labels.append(listrip.split()[0])                    
                    line=src.readline()
                    listrip = line.strip()                              
                while listrip != '':
                    entries.append(listrip.split())
                    l = src.readline()
                    listrip = l.strip()                  
                # if the block contains the set of atoms                                       
                if '_atom_site_fract_x' in labels:
                    magnetic_positions = []
                    atomlabels = {}
                    for i,l in enumerate(labels):
                        if l == "_atom_site_fract_x":
                            xcol = i
                        elif l == "_atom_site_fract_y":
                            ycol = i
                        elif l == "_atom_site_fract_z":
                            zcol = i
                        elif l == "_atom_site_type_symbol":
                            tcol = i
                        elif l == "_atom_site_label":
                            labelcol = i
                            
                    for i,entry in enumerate(entries):
                        if entry[tcol] in magnetic_atoms:
                            magnetic_positions.append(np.array([float(entry[xcol]),\
                                                                float(entry[ycol]),\
                                                                float(entry[zcol])]))
                            magnetic_species.append(entry[tcol])
                            atomlabels[entry[labelcol]] = i

                # If the block contains the set of bonds
                if '_geom_bond_atom_site_label_1' in labels:
                    jlabelcol = None
                    bondlists = []
                    bond_distances = []
                    bond_labels = {}
                    for i,l in enumerate(labels):                    
                        if l == "_geom_bond_atom_site_label_1":
                            at1col = i
                        if l == "_geom_bond_atom_site_label_2":
                            at2col = i
                        if l == "_geom_bond_distance":
                            distcol = i
                        if l == "_geom_bond_label":
                            jlabelcol = i
                    if jlabelcol is None:
                        for en in entries:
                            newbond = (atomlabels[en[at1col]],
                                       atomlabels[en[at2col]])

                            if en[distcol] not in bond_distances:
                                bond_distances.append(en[distcol])
                                bond_labels["J"+str(len(bond_distances))] = \
                                        len(bond_labels)
                                bondlists.append([])
                            bs = bond_distances.item(en[distcol])
                            bondlists[bs].append(newbond)
                    else:
                        for en in entries:
                            newbond = (atomlabels[en[at1col]],
                                       atomlabels[en[at2col]])
                            if bond_labels.get(en[jlabelcol]) is None:
                                bond_labels[en[jlabelcol]] = len(bondlists)
                                bond_distances.append(en[distcol])
                                bondlists.append([])
                            bondlists[bond_labels[en[jlabelcol]]].append(newbond)
                    bond_labels = sorted([(bond_labels[la],la) for la in bond_labels],key=lambda x:x[0])
                    bond_labels = [x[1] for x in bond_labels]
            # Build the Bravai's vectors

        bravais_vectors = []                                                        
        if bravais_params.get('a') is not None:
            bravais_vectors.append(np.array([bravais_params.get('a'),0,0]))
    
        if bravais_params.get('b') is not None:
            gamma = bravais_params.get('gamma')
            if gamma is None:
                gamma = 3.1415926*.5
            bravais_vectors.append(np.array([bravais_params.get('b')*np.cos(gamma),
                                        bravais_params.get('b')*np.sin(gamma),0]))
                
        if bravais_params.get('c') is not None:
            alpha = bravais_params.get('alpha')
            beta = bravais_params.get('beta')
            if alpha is None:
                alpha = 3.1415926*.5
            if beta is None:
                beta = 3.1415926*.5
            x = np.cos(alpha)
            y = np.cos(beta) - x * np.cos(gamma)
            y = y/np.sin(gamma)
            z = bravais_params.get('c') * np.sqrt(1-x*x-y*y)
            x = bravais_params.get('c') * x
            y = bravais_params.get('c') * y                           
            bravais_vectors.append(np.array([x,y,z]))

    magnetic_positions = np.array(magnetic_positions).dot(\
                                                    np.array(bravais_vectors))
    model= MagneticModel(magnetic_positions, bravais_vectors, 
                         bond_lists=bondlists,
                         bond_names=bond_labels, 
                         magnetic_species=magnetic_species )
    model.bond_distances = [float(d) for d in bond_distances]    
    return model






def magnetic_model_from_wk2_struct(filename, magnetic_atoms=["Mn","Fe","Co","Ni","Dy","Tb","Eu"],bond_names=None):    
    bravais_params = {}
    magnetic_positions = None
    bravais_vectors = None
    labels = None
    entries = None
    magnetic_species = []
    bond_labels = None
    bondlists = None
    bond_distances = []

    with open(filename) as fin:
        title = fin.readline()
        fin.readline() # size
        fin.readline() # not any clue
        bravais = fin.readline()
        for l in fin:
            sl = l.strip()
            if sl[:4] == "ATOM":
                positions = []
                fields = sl.split()
                positions.append([float(fields[2][3:]),float(fields[3][3:]),float(fields[4][3:])])
                print("ATOM:")
                print(fields)
                mult = int(fin.readline().strip().split()[1])
                for k in range(mult):
                    sl = fin.readline()
                    fields = sl.split()
                    print(fields)
                    positions.append([float(fields[1][3:]),float(fields[2][3:]),float(fields[2][3:])])
                    
                atomlabelfield = fin.readline().strip()
                if atomlabelfield[1] == " ":
                    atomlabel = atomlabelfield[0] + atomlabelfield[1]
                    atomspecies = atomlabelfield[0]
                else:
                    atomlabel = atomlabelfield[0]
                    atomspecies = atomlabelfield[:2]
                
                lrm = fin.readline()              #Rotation matrix
                lrm = lrm + fin.readline()
                lrm = lrm + fin.readline()

                if atomspecies not in magnetic_atoms:
                    break
                
                print(atomlabel, positions,mult)
            print(l)

        bravais_fields = bravais.strip().split()
        bravais_params["a"] = bravais_fields[0]
        bravais_params["b"] = bravais_fields[1]
        bravais_params["c"] = bravais_fields[2]
        bravais_params["alpha"] = bravais_fields[3]*3.1415926/180
        bravais_params["beta"] = bravais_fields[4]*3.1415926/180
        bravais_params["gamma"] = bravais_fields[5]*3.1415926/180
        

        bravais_vectors = []                                                        
        if bravais_params.get('a') is not None:
            bravais_vectors.append(np.array([bravais_params.get('a'),0,0]))
    
        if bravais_params.get('b') is not None:
            gamma = bravais_params.get('gamma')
            if gamma is None:
                gamma = 3.1415926*.5
            bravais_vectors.append(np.array([bravais_params.get('b')*np.cos(gamma),
                                        bravais_params.get('b')*np.sin(gamma),0]))
                
        if bravais_params.get('c') is not None:
            alpha = bravais_params.get('alpha')
            beta = bravais_params.get('beta')
            if alpha is None:
                alpha = 3.1415926*.5
            if beta is None:
                beta = 3.1415926*.5
            x = np.cos(alpha)
            y = np.cos(beta) - x * np.cos(gamma)
            y = y/np.sin(gamma)
            z = bravais_params.get('c') * np.sqrt(1-x*x-y*y)
            x = bravais_params.get('c') * x
            y = bravais_params.get('c') * y                           
            bravais_vectors.append(np.array([x,y,z]))

    magnetic_positions = np.array(magnetic_positions).dot(\
                                                    np.array(bravais_vectors))
    model= MagneticModel(magnetic_positions, bravais_vectors, 
                         bond_lists=None,
                         bond_names=None, 
                         magnetic_species=magnetic_species,
                         model_label=title)
    model.bond_distances = [float(d) for d in bond_distances]    
    return model




def map_config_model1_model2(model1,config,model2,tol=.1):
    if len(model1.coord_atomos)>len(model2.coord_atomos):
        print("alert: unit cell in model2 is smaller than in model1.")
    if len(model1.coord_atomos)!=len(config):
        print("size of model1 is different that length of the configuration")

    size1 = len(model1.coord_atomos)
    dictatoms = [-1 for p in model2.coord_atomos]
    for i,p in enumerate(model2.coord_atomos):
        for j,q in enumerate(model1.supercell):
            if np.linalg.norm(p-q)<tol:
                dictatoms[i] = j % size1
                break

    if config[0] is list:
        return [[c[j] for j in dictatoms] for c in configs]
    else:
        return [config[j] for j in dictatoms]

            
    
    
    



