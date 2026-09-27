// Compileable integration placeholder for a 2x2 mesh NoC.
// Replace this module with the real RTL, then align the Python placeholder
// signal names in cocotb_tb/{noc_interface,traffic_driver}.py to its port list.
module noc_2x2 (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       packet_valid_i,
    output logic       packet_ready_o,
    input  logic [1:0] packet_src_i,
    input  logic [1:0] packet_dest_i,
    input  logic       packet_type_i,
    input  logic [2:0] packet_length_i,
    input  logic       packet_vc_i,
    input  logic       packet_priority_i,
    input  logic       packet_sop_i,
    input  logic       packet_eop_i,
    input  logic [1:0] packet_flit_type_i
);
    // Always-ready stub so the cocotb integration path is runnable immediately.
    assign packet_ready_o = rst_n;
endmodule
