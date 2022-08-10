NVCC = nvcc
CFLAG = -O3
FFLAG = -Mcuda=cc70 -fast
NVCCFLAG = -arch=sm_70 -Xcompiler=-O3

dummy:
	@echo -e "make cuda\t\tcompile all CUDA sources"
	@echo -e "make clean\t\trm -f *.x *.mod"
		
cuda: binary.cu

binary.cu: *.cu
	$(NVCC) $(NVCCFLAG) $< -o $@

clean:
	rm -f *.x *.mod *.ptx



