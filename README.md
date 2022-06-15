# MastersThesisCodes
An Efficient 3D-model for the Micro-Scale Analysis of Geometrically Imperfect Fibre-Reinforced Composites

The final code iteration or best attempt for three different approaches for the efficient microscale modelling of FRPs:

- Carrera Unified Formulation (best attempt)
  - CUFclean_TE_LinBeam.py : Taylor Expansion Model CUF solver
  - inputdatTE.py : Input code for generating YAML data file for Taylor Expansion model
  - dataTE.yaml : Data file with mesh, geometry, and material data for Taylor Expansion Model
  - CUF_LE_9GP.py : Lagrange Expansion Model CUF solver
  - inputquadLE.py : Input code for generating the YAML data file for Lagrange Expansion Model
  - dataLE.yaml : Data file with mesh, geometry, and material data for Lagrange Expansion Model
  
- Geometrically Exact IGA (best attempt)
  - SplineIGA_Halton.py : Complete code attempt for the Geom. Exact Isogeometric beam including Halton sampling and regression
  
- Final Unit Cell Model (Complete and Fully functional)
  - ExternalMesher_Multicell_Quadratic.py : External Meshing Routine
  - InternalMesher_Multicell_Quadratic.py : Internal Meshing Routine
  - MainRoutine_Perturbation.py : Main routine for running perturbed fibre simulations
  - MainRoutine_Validation.py : Main routine for running validation simulations for Global Material Properties
  - Perturbation_Random.py : Perturbation algorithm
  - Wedge_Quadratic_UnitCell_Solver.py : Solver for Wedge element Unit Cell
  - WedgePost.py : Post Processing routines
