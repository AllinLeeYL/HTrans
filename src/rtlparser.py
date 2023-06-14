from pyverilog.vparser.parser import parse
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx
import torch, re, os
import torch_geometric as pyg

def preprocess(files:list, 
               outPath:str):
    dirname = os.path.dirname(files[0])
    combined_lines = []
    included_files = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if re.match(r'`include', line): # TODO: only include once -> nested include
                included_file = line[line.index('`include')+8:].split('\"')[1]
                if included_file not in included_files:
                    with open(os.path.join(dirname, included_file), 'r', encoding='utf-8') as f:
                        combined_lines += f.readlines()
                    included_files.append(included_file)
                    continue
                else:
                    continue
            combined_lines.append(line)
    with open(outPath, 'w', encoding='utf-8') as f:
        f.writelines(combined_lines)

class AST:
    def __init__(self, verilog_files) -> None:
        self.ast, _ = parse(verilog_files)
        self.ast.calculate()
        self.type = None
        self.DICTIONARY_GEN = \
            ["Source","Description","Ioport","Decl","Lvalue"]
        self.ARRAY_GEN = \
            ["ModuleDef","Paramlist","Portlist","Input","Width","Reg","Wire","Rvalue","ParseSelect",
             "Uplus","Uminus","Ulnot","Unot","Uand","Unand","Uor","Unor","Uxnor","Power","Times","Divide","Mod","Plus",
             "Minus","Sll","Srl","Sla","Sra","LessThan","GreaterThan","LessEq","GreaterEq","Eq","Eql","NotEq","Eql","NotEql",
             "And","Xor","Xnor", "Uxor", "Or","Land","Lor","Cond","Assign","Always","AlwaysFF","AlwaysComb","AlwaysLatch",
             "SensList","Sens","Substitution","BlockingSubstitution","NonblockingSubstitution","IfStatement","Block",
             "Initial","Plus","Output","Partselect","Port","InstanceList","Instance","PortArg","Pointer","Concat", "Parameter", 
             "SystemCall", "CaseStatement", "Case", "Function", "CasezStatement", "FunctionCall", "Dimensions", "Length", 
             "LConcat", "Concat", "SingleStatement", "Repeat", "Integer", "CasexStatement", "ForStatement", "Localparam",
             "EventStatement", "DelayStatement", "Task", "ParamArg", "Inout"]
        self.CONST_DICTIONARY_GEN = \
            ["IntConst","FloatConst","StringConst","Identifier"]
    def toNetworkX(self):
        if hasattr(self, 'nx'):
            return self.nx
        self.kinds = self.DICTIONARY_GEN + self.ARRAY_GEN + self.CONST_DICTIONARY_GEN
        self.nx = self.ast.toNetworkX()
        for node in self.nx.nodes.data():
            attr = node[1]
            rp = attr['rp'] if 'rp' in attr.keys() else 1.0
            ap = attr['ap'] if 'ap' in attr.keys() else 1.0
            node[1]['x'] = torch.tensor([node[0],
                                         attr['lineno'],
                                         self.kinds.index(attr['kind']),
                                         rp,
                                         ap],
                                         dtype=torch.float)
        return self.nx
    def toPyG(self):
        if hasattr(self, 'pyg'):
            return self.pyg
        self.pyg = pyg.utils.from_networkx(self.toNetworkX())
        return self.pyg
    def show(self):
        self.ast.show()