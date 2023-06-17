1.Home directory includes:
 

1.1 src
 
	--- Verilog codes implemeting PIC16F84

1.2 PIC16F84-T300.pdf 


2.Trojan
  
Trojan Description
  	
	The Trojan trigger, a state machine, observes the number of execution of specific instruction. 
	Above a certain number of execution the Trojan is triggered, and it manipulates address lines to the external EEPROM and changes to address 0.


Trojan Taxonomy

	Insertion phase: Design
	Abstraction level: Register-transfer level 
	Activation mechanism: Internally conditionally triggered
	Effects: Denial of Service
	Location: Processor
	Physical characteristics: Functional