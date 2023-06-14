import pickle, rtlparser, dataloader, os

def test1():
    basedir = '/home/allin/HTrans/datasets/'
    recipeloader = dataloader.RecipeLoader(basedir)
    astloader = dataloader.ASTLoader(recipeloader)
    for ast in astloader:
        if ast.name == 'AES-T400':
            nx = ast.toNetworkX()
            for node in nx.nodes.data():
                print(node)

if __name__ == '__main__':
    test1()
    