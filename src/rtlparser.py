from pyverilog.vparser.parser import parse
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx
import torch, re, os
import torch_geometric as pyg
import numpy as np

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
        print('processing:', verilog_files[0])
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
        maxlineno = np.max([n[1]['lineno'] for n in self.nx.nodes.data()])
        for node in self.nx.nodes.data():
            attr = node[1]
            rp = attr['rp'] if 'rp' in attr.keys() else 1.0
            ap = attr['ap'] if 'ap' in attr.keys() else 1.0
            node[1]['x'] = torch.tensor([node[0] / len(self.nx.nodes.data()),
                                         attr['lineno'] / maxlineno,
                                         self.kinds.index(attr['kind']) / len(self.kinds),
                                         rp,
                                         ap],
                                         dtype=torch.float)
        return self.nx
    def toPyG(self):
        if hasattr(self, 'pyg'):
            return self.pyg
        self.pyg = pyg.utils.from_networkx(self.toNetworkX())
        return self.pyg
    # def TjLoc_Feature(self):
    #     if hasattr(self, 'tjfeature'):
    #         return self.tjfeature
    #     if not hasattr(self, 'nx'):
    #         self.toNetworkX()
    #     count = 0
    #     for node in self.nx.nodes.data():
    #         attr = node[1]
    #         for token in self.TjLoc:
    #             if attr['token'] == token:
    #                 count += 1
    #                 if hasattr(self, 'tjfeature'):
    #                     self.tjfeature += node[1]['x']
    #                 else:
    #                     self.tjfeature = node[1]['x']
    #                 break
    #     if count >= len(self.TjLoc):
    #         self.tjfeature = self.tjfeature / count
    #         self.tjfeature = torch.reshape(self.tjfeature, (1, 5))
    #         return self.tjfeature
    #     else:
    #         print(self.name, 'has not identifier', self.TjLoc)
    #         raise Exception
    def TjLoc_Feature(self):
        if hasattr(self, 'tjfeature'):
            return self.tjfeature
        if not hasattr(self, 'nx'):
            self.toNetworkX()
        count = 0
        for node in self.nx.nodes.data():
            attr = node[1]
            if attr['token'] == self.TjLoc[0]:
                self.tjfeature = node[1]['x']
                self.tjfeature = torch.reshape(self.tjfeature, (1, 5))
                return self.tjfeature
        print(self.name, 'has not identifier', self.TjLoc)
        raise Exception
    def find_k_nearest(self, pred, lossfunc, k=5):
        nodes = []
        for node in self.nx.nodes.data():
            attr = node[1]
            if attr['x'][4] > 0.2 or attr['kind'] != 'Identifier':
                continue
            loss = float(lossfunc(pred, torch.reshape(attr['x'], (1, 5))))
            if len(nodes) < k and attr['token'] not in [node[0][1]['token'] for node in nodes]:
                nodes.append([node, loss])
            elif loss < np.max([n[1] for n in nodes]) and attr['token'] not in [node[0][1]['token'] for node in nodes]:
                nodes[-1] = [node, loss]
            nodes.sort(key=lambda a: a[1], reverse=False)
        return [node[0] for node in nodes]
    def show(self):
        self.ast.show()