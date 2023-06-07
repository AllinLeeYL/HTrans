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
    def toNetworkX(self):
        pass
    def toPyG(self):
        pass
    def show(self):
        self.ast.show()