
BAMTOOLS= $(realpath ../bamtools/)
SAMTOOLS= $(realpath ../samtools/)
HTSLIB= $(realpath ../htslib/)


#defined via user
ifdef BAMTOOLSINC
	BAMTOOLSINCAPI=${BAMTOOLSINC}
	BAMTOOLSINCSHARED=${BAMTOOLSINC}
else
	BAMTOOLSINCAPI=${BAMTOOLS}/build/src/
	BAMTOOLSINCSHARED=${BAMTOOLS}/src/
endif

ifdef BAMTOOLSLIB
	BAMTOOLSLIBOBJ=${BAMTOOLSLIB}/libbamtools.a 
else
	BAMTOOLSLIBOBJ=${BAMTOOLS}/build/src/libbamtools.a 
endif

ifeq ($(CXX),)
CXX := g++ #-g  -pg 
endif

ifdef CONDA_PREFIX
	CXXFLAGS += -I$(CONDA_PREFIX)/include
	LDFLAGS  += -L$(CONDA_PREFIX)/lib
endif

ifeq ($(SAMTOOLSINC),)#no defined 
SAMTOOLSINC := ${SAMTOOLS}
endif

ifeq ($(HTSLIBINC),)#no defined 
HTSLIBINC := ${HTSLIB}
endif

empty :=
space := $(empty) $(empty)
escape_path = $(subst $(space),\$(space),$1)

BAMTOOLSINCAPI_ESC     := $(call escape_path,$(BAMTOOLSINCAPI))
BAMTOOLSINCSHARED_ESC  := $(call escape_path,$(BAMTOOLSINCSHARED))
SAMTOOLSINC_ESC        := $(call escape_path,$(SAMTOOLSINC))
HTSLIBINC_ESC          := $(call escape_path,$(HTSLIBINC))
BAMTOOLSLIBOBJ_ESC     := $(call escape_path,$(BAMTOOLSLIBOBJ))

CXXFLAGS += -Wall -O3 -I. -I$(BAMTOOLSINCAPI_ESC) -I$(BAMTOOLSINCSHARED_ESC)
LDLIBS   += -lz
LDLIBSHTS = $(SAMTOOLSINC_ESC)/bedidx.o $(HTSLIBINC_ESC)/libhts.a $(SAMTOOLSINC_ESC)/libst.a
CXXFLAGSHTS = -I$(HTSLIBINC_ESC) -I$(SAMTOOLSINC_ESC)

#.PHONY: clean all 

all: libgab.o testUtils targetTest PutProgramInHeader.o PutProgramInHeaderHTS.o gzstream/gzstream.o FastQObj.o FastQParser.o libgab.a

BamFunctions.o:
	${CXX} ${CXXFLAGS} ${CXXFLAGSHTS} -c -o BamFunctions.o BamFunctions.cpp
	ar cr libgabbamhts.a BamFunctions.o

gzstream/libgzstream.a:
	make CXX=${CXX} CPPFLAGS="${CXXFLAGS}" -C gzstream/

gzstream/gzstream.o:
	make CXX=${CXX} CPPFLAGS="${CXXFLAGS}" -C gzstream/

testUtils:	testUtils.o  libgab.a gzstream/libgzstream.a
	${CXX} $(LDFLAGS) -o $@ $^ $(LDLIBS)

targetTest:	targetTest.o libgab.a gzstream/libgzstream.a
	${CXX} $(LDFLAGS) -o $@ $^ $(LDLIBS)

testRecons:	testRecons.o libgab.a $(BAMTOOLSLIBOBJ_ESC) ReconsReferenceBAM.o gzstream/libgzstream.a
	${CXX} $(LDFLAGS) -o $@ $^ $(LDLIBS)

libgab.a: libgab.o FastQObj.o FastQParser.o PutProgramInHeaderHTS.o
	ar cr libgab.a libgab.o FastQObj.o FastQParser.o PutProgramInHeaderHTS.o

test:	all
	./test.sh

%.o: %.cpp
	${CXX} ${CXXFLAGS} -c $^ -o $@

clean :
	rm -f testRecons testUtils *.o  *.a
	make  -C gzstream/ clean

.PHONY: all
