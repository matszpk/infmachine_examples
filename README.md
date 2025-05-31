## Examples for the Infinite Machine.

There are examples for the Infinite Machine. It provides simple programs that
generates machines and data that machines.

### Program prefix_op

Program prefix_op generates machine that make prefix operation.
To generate machine enter:

```
prefix_op machine OP CELL_LEN_BITS DATA_PART_LEN PROC_NUM [MAX_PROC_NUM_BITS]
```

Arguments are:
* `OP` - operation `add` for prefix_sum, `mul1` for prefix_product, `and`, `or`,
`xor`, `min` and `max`.
* CELL_LEN_BITS - power of two of memory cell length. CELL_LEN_BITS=3, then cell length is 8 bits.
* DATA_PART_LEN - internal data part length.
* PROC_NUM - number of processors.
* MAX_PROC_NUM_BITS - maximal number of power of two number of processors.

File of machine will be printed to standard output in TOML file format.

To generate data and expected output data run:

```
prefix_op data_and_exp OP CELL_LEN_BITS DATA_PART_LEN PROC_NUM MAX_VALUE DATA_PATH EXP_PATH
```

* CELL_LEN_BITS - power of two of memory cell length. CELL_LEN_BITS=3, then cell length is 8 bits.
* DATA_PART_LEN - internal data part length.
* PROC_NUM - number of processors.
* MAX_VALUE - maximal value for entry value in data.
* DATA_PATH - path to data file
* EXP_PATH - path to expected output data.

### Program simple_inc, simple_inc2 and simple_inc3.

Program generates machine that make simple incrementation of values.
To generate machine enter:

```
simple_inc CELL_LEN_BITS PROC_NUM_BITS [REAL_PROC_NUM_BITS]
```

* CELL_LEN_BITS - power of two of memory cell length. CELL_LEN_BITS=3, then cell length is 8 bits.
* PROC_NUM_BITS - power of two of number of processors.
* REAL_PROC_NUM_BITS - power of two of number of processors. For environment setup.

File of machine will be printed to standard output in TOML file format.

### Program state_test.

See to sources to find out how to use that test.

### Program utils_test.

Program that tests utils module that simplify some operations and provides some simple
boilerplate routines.
See to source to find out about how to run that program.

### Util module.

This module provides some simple routines to manage data.

Mainly, this module provides routine to manage stage of calculations. Machine program is
built from stages that can do some operations defined by user.

See to sources of prefix_op and utils_test to find out about how to use that module.
See to sources to find out about that module.
