############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project hls_miniProject
set_top ppg_rf_infer
add_files ppg_rf_hls/firmware/BDT.cpp
add_files ppg_rf_hls/firmware/ppg_rf_project.cpp
open_solution "solution1" -flow_target vivado
set_part {xc7a100tcsg324-1}
create_clock -period 10 -name default
config_cosim -tool xsim
#source "./hls_miniProject/solution1/directives.tcl"
#csim_design
csynth_design
#cosim_design
export_design -rtl verilog -format ip_catalog
