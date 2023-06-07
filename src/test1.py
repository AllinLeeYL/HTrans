import pickle, rtlparser, dataloader, os

def test1():
    basedir = '/home/allin/HTrans/datasets/'
    recipeloader = dataloader.RecipeLoader(basedir)
    astloader = dataloader.ASTLoader(recipeloader)
    for ast in astloader:
        if ast.name == 'AES-T400':
            ast.ast.calculate()
            ast.show()

if __name__ == '__main__':
    test1()
    