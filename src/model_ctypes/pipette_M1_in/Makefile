CC=gcc
CFLAGS=-Wall -fPIC -c
EXECUTABLE=ina.so
SOURCES=ina.c run.c
OBJECTS=$(SOURCES:.c=.o)

all:$(EXECUTABLE)
$(EXECUTABLE):$(OBJECTS)
	$(CC) -shared $(OBJECTS) -o $(EXECUTABLE) -lm  -L/data90t/users/abrasheva/WORK/INa_full_trace/env_pypoptim/include/lib -lsundials_cvode

ina:ina.c
	$(CC) $(CFLAGS) -I/data90t/users/abrasheva/WORK/INa_full_trace/env_pypoptim/include ina.c
	$(CC) $(CFLAGS) -I/data90t/users/abrasheva/WORK/INa_full_trace/env_pypoptim/include run.c

clean:
	rm -rf $(OBJECTS) $(EXECUTABLE)
