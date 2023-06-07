`timescale 1ns / 1ps

module top(clk, rst, state, key, out);
    input          clk, rst;
    input  [127:0] state, key;
    output [127:0] out;

		aes_128 AES (clk, state, key, out);
		Trojan_Trigger Trigger(rst, clk, state, Tj_Trig);
		TSC Trojan (clk, rst, key, Tj_Trig);

endmodule
