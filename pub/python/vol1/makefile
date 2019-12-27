
all: ft01 ft02 ft03 ft04 ft05 ft06 ft07 ft08 ft09 ft12

nonfunctioning: ft10 ft11

ft01:
	python3 ft01_poisson.py

ft02:
	python3 ft02_poisson_membrane.py

ft03:
	python3 ft03_heat.py

ft04:
	mpirun -n 8 python3 ft04_heat_gaussian.py

ft05:
	python3 ft05_poisson_nonlinear.py

ft06:
	python3 ft06_elasticity.py

ft07:
	python3 ft07_navier_stokes_channel.py

ft08:
	python3 ft08_navier_stokes_cylinder.py

ft09:
	python3 ft09_reaction_system.py

ft10:
	python3 ft10_poisson_extended.py

ft11:
	python3 ft11_magnetostatics.py

ft12:
	python3 ft12_poisson_solver.py
