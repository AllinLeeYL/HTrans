1.Home directory includes:

1.1 src
	--- Source files for both Trojan free and Trojan inserted designs

2.Trojan
Trojan Description
	The Trojan Trigger is a state machine based on the output data of the first master. 
	At a specific state the Trojan changes the priority of the first master to the highest value in all slaves 
	by manipulation their corresponding configuration registers in the register file. 

Trojan Taxonomy
	Insertion phase: Design
	Abstraction level: Register transfer level
	Activation mechanism: Internally conditionally triggered
	Effects: Change Functionality, Denial of Service
	Location: I/O
	Physical characteristics: Functional