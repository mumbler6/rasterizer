.PHONEY: build, run

build: 

run:	
	python3 main.py $(file)

clean:
	rm -rf __pycache__
	rm -rf venv
