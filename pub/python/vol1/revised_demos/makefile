all: poisson eigenvalue cahn-hilliard

poisson:
	python3 demo_poisson.py

nonlinear_poisson:
	python3 demo_nonlinear-poisson.py

eigenvalue:
	python3 demo_eigenvalue.py

cahn-hilliard:
	python3 demo_cahn-hilliard.py

c00.xmk.gz c01.xmk.gz c11.xmk.gz mesh.xml.gz:
	python3 generate_data.py


