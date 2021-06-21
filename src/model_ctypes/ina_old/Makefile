
SRCS =./ina_old.c \
./run_old.c

OBJS = $(SRCS:%.c=%.o)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS-add) $(CFLAGS) $(fPIC) -DNDEBUG -c $< -o $@

#release debug: ina_old.$(SHLIB_EXT)

ina_old.so: $(OBJS)
	rm -rf $@
	$(CC) $(LDFLAGS) $(LDFLAGS-add) -shared -o $@ $^
