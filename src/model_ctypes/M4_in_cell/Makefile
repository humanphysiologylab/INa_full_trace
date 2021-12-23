CC=gcc
CFLAGS=-Wall -fPIC -c
EXECUTABLE=ina.so
SOURCES=ina.c run.c
OBJECTS=$(SOURCES:.c=.o)

all:$(EXECUTABLE)
$(EXECUTABLE):$(OBJECTS)
	$(CC) -shared $(OBJECTS) -o $(EXECUTABLE) -lm  -L/home/nik/miniconda3/envs/env_pypoptim/lib -lsundials_cvode

ina:ina.c
	$(CC) $(CFLAGS) -I/home/nik/miniconda3/envs/env_pypoptim/include ina.c
	$(CC) $(CFLAGS) -I/home/nik/miniconda3/envs/env_pypoptim/include run.c

clean:
	rm -rf $(OBJECTS) $(EXECUTABLE)
