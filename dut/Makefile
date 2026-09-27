# Set SIM=verilator to use Verilator instead of the default Icarus Verilog.
SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES := $(PWD)/noc_2x2.sv
TOPLEVEL := noc_2x2
MODULE := training.cocotb_test_entry
export PYTHONPATH := $(PWD)/..:$(PYTHONPATH)

include $(shell cocotb-config --makefiles)/Makefile.sim
