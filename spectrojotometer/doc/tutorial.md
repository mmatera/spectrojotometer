Spectrojotometer.
==================

This package is a set of tools to determine, by means of ab initio 
simulations, the value of the coupling constants in a Heisenberg-like effective 
magnetic model for a given magnetic system.


Instalation:
============

To install the package, just run

python3 setup.py install


Loading models: conventional vs. primitive cell
=================================================

`magnetic_model_from_file` / `magnetic_model_from_cif` accept an optional
`primitive_cell` parameter (default `False`). See the main README for
details; in short, `primitive_cell=True` keeps only the atoms declared
in the CIF's asymmetric unit (e.g. 4 instead of 16 for ZnCr2O4) and
switches the model's Bravais vectors to a primitive basis, instead of
expanding atoms to fill the conventional cell.



Visual Bond
===========

Visual Bond is a graphic interfase for the tools in the spectrojotometer package.

![First step: load the magnetic model](img/img-00.png)


![First step: load the magnetic model](img/img-01.png)



![First step: load the magnetic model](img/img-06.png)


Estimating parameter uncertainties
==================================

The DFT energies are only known within a tolerance δE (the *Energy
tolerance* field). The couplings compatible with the data are those that
reproduce **every** computed energy within that tolerance. They form a
convex polytope in the space of parameters, the *compatibility polytope*
(green in the figure). The value reported for each coupling is the
center of the smallest box that contains the polytope, and its error bar
is the half width of the box (dashed green: *uncertainty in parameters*).

![Estimation of compatibility regions and parameter uncertainties](img/regions.svg)

The polytope always lies between two ellipses centered at the
least-squares solution: a lower bound (blue), contained in it, and an
upper bound (red), which contains it. The *Error bound method* selector
chooses how the box is obtained:

* **Exact box (linear programming)**, the default: the exact box of the
  polytope, obtained by minimizing and maximizing each coupling over it.
* **Monte Carlo (inner estimate)**: points are drawn uniformly inside the
  upper-bound ellipse, and only those inside the polytope are kept. The
  box of the kept points is always inside the exact box, and it
  underestimates it more as the number of couplings grows. The
  acceptance rate (the fraction of points kept) is printed with the
  results. *Num. samples* sets the number of points; leave *Size factor*
  at 1.
* **Quadratic bound (outer estimate)**: the box of the upper-bound
  ellipse (dashed red). It always contains the exact box.

The results also list *region constraints*: inequalities on
combinations of couplings, `|v·J − c| < δ`, which are often much tighter
than the individual error bars, because the polytope is usually
elongated.

If no set of couplings reproduces all the energies within the tolerance
(the polytope is empty), the couplings are marked as *incompatible*:
either the tolerance is too small, or the model lacks some relevant
coupling. A coupling that the configurations with known energies do not
fix is shown as *undetermined*: add configurations that involve it.







Terminal Tools in the package and workflow:
============================================



The package consists of these three command line programs:


*) bond_generator: 

   From an standard cif file, produces a new cif file containing information
   about the bonds that defines a Heisenberg magnetic effective model. If the
   input file has already information about bonds, the output file contains both
   the bonds in the input as well as the new generated bonds.


   usage: bond_generator [-h] [--discretization [value]] [--rmin [value]]
                         [--rmax [value]]
                         ciffile outfile


   positional arguments:
     ciffile               load atomic positions from a cif file
     outfile               Name of the outputfile. If it is not provided, the
                        output is printed to the standard output.

   optional arguments:
     -h, --help            show this help message and exit
     --discretization [value]
                             the tolerance in considering two distances as the
                             same.
     --rmin [value]        the minimal distance between atoms of the bond
     --rmax [value]        the maximal distance between atoms of the bond


As an example, by run 

   bond_generator.py  --rmin 0 --rmax 4.9 --discretization .02  example1.cif example1-nnn.cif

   for the  example1.cif  file in the examples folder, example1-nnn.cif file is created. The model can be modified by hand
   by means of a plain text editor.


*) optconfs
    Generate spin configurations to determine the coupling constants of a model
    with minimal condition number
    
    
    usage: optconfs.py [-h] [--known [known configuration file]]
                       [--start [start file]] [-n numconfs] [-u updatesize]
                       [--it iterations]
                       ciffile outfile
    
    
    positional arguments:
      ciffile               load atomic positions from a cif file
      outfile               Name of the outputfile. If it is not provided, the
                            output is printed to the standard output.
    
    optional arguments:
      -h, --help            show this help message and exit
      --known [known configuration file]
                            file with the set of configurations that we want to
                            increase
      --start [start file]  A file containing an initial guess for the target
                            configurations.
      -n numconfs           number of new configurations to be generated
      -u updatesize         number of random configurations to be generate on each
                            step
      --it iterations       number of iterations
    
    
As an example, by run

   optconfs  example1-nnn.cif optconfs1.out
   
   on the examples/example1-nnn.cif generated by bond_generator, the file optconfs1.out is created, containing a list
   of magnetic configurations from which energies the coefficient couplings can be estimated. The program also writes
   a recovery file example1-nnn-resume.tmp, which can be used to improve the magnetic configuration choice:

   optconfs  --start example1-nnn-resume.tmp example1-nnn.cif optconfs1.out

   If we want to know what would be most profiteable magnetic configurations to be simulated once we have already simulated 
   a given set of configurations, we run

   optconfs  --known optconfs1.out example1-nnn-resume.tmp example1-nnn.cif optconfs2.out

   In this case, the program will try to provide the optimal configuration to be added to those in optconfs1.out.
   We can ask for an specific amount of new configurations throw the parameter -n.


*) evaluate_cc

    From a set of energies and magnetic configurations, estimates the
    coupling constants of a model, with error bars given by a box around
    the compatibility region (see "Estimating parameter uncertainties"
    above).

    usage: evaluate_cc [-h] [-t [value]] [-m {exact,montecarlo,ellipse}]
                       [--mcsteps [value]] [-e] [-x]
                       modelfile [input file]

    From a set of energies and magnetic configurations, estimates the coupling
    constants of a model.

    positional arguments:
      modelfile             load atomic positions from a model file.
      [input file]          file containing evaluated energies and magnetic
                            configurations.

    options:
      -h, --help            show this help message and exit
      -t [value], --tolerance [value]
                            Provides the estimated error for the energies.
      -m {exact,montecarlo,ellipse}, --method {exact,montecarlo,ellipse}
                            How to bound the compatibility region: exact box by
                            linear programming (default), Monte Carlo sampling
                            (inner estimate) or the bounding box of an ellipsoid
                            containing it (outer estimate).
      --mcsteps [value]     Number of Monte Carlo samples (only with --method
                            montecarlo).
      -e, --equations       Prints the resulting equations.
      -x, --showchi         Prints the list of errors in the predicted
                            configuration energy.

    In the input file, each line consists of the energy of a magnetic
    configuration, followed by the configuration as a list [0,1,0,...].
    Anything after a # is considered a comment. Configurations whose
    energy is nan are ignored. For example:

    # Energy  Configuration  Label
    143121.12   [0,0,0,0,0,0,0]  # Ferro up
    143121.02   [1,1,0,0,1,1,1]  # Disorder1
    143121.24   [1,0,1,1,1,1,0]  # Disorder2
    143121.12   [1,1,1,1,1,1,1]  # Ferro down

    Running, with the files in the examples folder,

    evaluate_cc examples/example1.cif examples/example1.spin

    prints (after some diagnostic messages)

    # Results (method: exact):
    ===========
    <E> =  49022.002225000004  (average energy of the configurations)
    J0  = ( -1.0  +/- 2.2222222168326238 ) *  0.00022499999977299012
    max relative error: 0.03333334461785853  in configuration  1

    Each coupling is shown as (value +/- error) * scale, where the scale
    is the largest coupling in absolute value. <E> is the average energy
    of the configurations. Here the three
    configurations do not determine J0 beyond the tolerance (the error
    bar is larger than the value): more configurations, or a smaller
    tolerance, are needed.

    With -x, the program also prints, for each configuration, the
    difference between the input energy and the one predicted by the
    model, relative to the tolerance. Once enough configurations have
    been evaluated, this checks the validity of the effective model:
    differences larger than 1 mean that the model cannot reproduce the
    input spectrum. If no set of couplings reproduces all the energies
    within the tolerance, the couplings are marked as (incompatible).



   



