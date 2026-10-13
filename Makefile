MODULE_TOPDIR = ../..

PGM = i.sam3.opencl

# CLIP's tokenizer vocabulary, read at run time (see LICENSE.bpe).
ETCFILES = bpe_simple_vocab_16e6.txt.gz

LIBES = $(IMAGERYLIB) $(VECTORLIB) $(DBMILIB) $(RASTERLIB) $(GISLIB) \
	$(MATHLIB)
DEPENDENCIES = $(IMAGERYDEP) $(VECTORDEP) $(DBMIDEP) $(RASTERDEP) \
	$(GISDEP)
EXTRA_INC = $(VECT_INC) $(OCLINCPATH) $(OPENMP_INCPATH) $(ZLIBINCPATH) -I$(OBJDIR)
EXTRA_CFLAGS = $(VECT_CFLAGS) $(OPENMP_CFLAGS) -O3 -std=gnu11
EXTRA_LIBS = $(OCLLIBPATH) $(OCLLIB) $(OPENMP_LIBPATH) $(OPENMP_LIB) -lcurl \
	$(ZLIBLIBPATH) $(ZLIB)

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd

# Embed the OpenCL kernels as a C string literal.
$(OBJDIR)/ocl_backend.o: $(OBJDIR)/sam3_kernels_cl.h

$(OBJDIR)/sam3_kernels_cl.h: sam3_kernels.cl | $(OBJDIR)
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' $< > $@
