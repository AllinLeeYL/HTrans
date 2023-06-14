import yaml, torch
import rtlparser, os

class DataLoader:
    def __init__(self) -> None:
        self.contents = []
    def __len__(self) -> int:
        return len(self.contents)
    def __getitem__(self, i):
        return self.contents[i]
    def append(self, item):
        self.contents.append(item)

class RecipeLoader:
    "load recipes and do some basic preprocess work"
    def __init__(self, path: str, preprocess: bool=True) -> None:
        self.preprocess = preprocess
        self.recipes = self.loadRecipesFromPath(path)
    def __getitem__(self, i:int):
        return self.recipes[i]
    def __len__(self) -> int:
        return len(self.recipes)
    def loadRecipesFromPath(self, path:str):
        self.path = path
        recipes = []
        for item in os.listdir(path):
            if os.path.isdir(os.path.join(path, item)):
                try:
                    with open(os.path.join(path, item, 'recipe.yaml'), 'r', encoding='utf-8') as f:
                        recipe = f.read()
                    recipe = yaml.load(recipe, Loader=yaml.CLoader)
                    # relative path -> absolute path, combine files
                    recipe = self.processRecipe(recipe, os.path.join(path, item))
                    recipes.append(recipe)
                except:
                    continue
        return recipes
    def processRecipe(self, recipe:dict, dirname:str):
        "relative path -> abspath, combine files into single one"
        fileBaseName = 'combined.v'
        TjfileBaseName = 'combined_tj.v'
        if 'TjFree' in recipe.keys():
            recipe['TjFree'] = [os.path.join(dirname, file) for file in recipe['TjFree']]
            if self.preprocess:
                filename = os.path.join(dirname, fileBaseName)
                rtlparser.preprocess(recipe['TjFree'], filename)
                recipe['TjFree'] = [filename]
        if 'TjIn' in recipe.keys():
            recipe['TjIn'] = [os.path.join(dirname, file) for file in recipe['TjIn']]
            if self.preprocess:
                filename = os.path.join(dirname, TjfileBaseName)
                rtlparser.preprocess(recipe['TjIn'], filename)
                recipe['TjIn'] = [filename]
        return recipe
    
class ASTLoader:
    "parse verilog code into AST"
    def __init__(self, recipes: RecipeLoader) -> None:
        self.recipes = recipes
        self.recipe2ast(recipes)
    def __getitem__(self, i:int):
        return self.asts[i]
    def __len__(self) -> int:
        return self.asts
    def recipe2ast(self, recipes:RecipeLoader):
        "accept RecipeLoader as input, construct ast from it"
        self.asts = []
        for recipe in recipes:
            # if 'MEMCTL' not in recipe['name']:
            #     continue
            if 'TjFree' in recipe.keys():
                ast = rtlparser.AST(recipe['TjFree'])
                ast.name = recipe['name']
                ast.label = 'TjFree'
                self.asts.append(ast)
            if 'TjIn' in recipe.keys():
                ast = rtlparser.AST(recipe['TjIn'])
                ast.name = recipe['name']
                ast.label = 'TjIn'
                if 'TjLoc' in recipe.keys():
                    ast.TjLoc = recipe['TjLoc']
                self.asts.append(ast)

def divide(astloader: ASTLoader, 
           devname: str = None,
           ratio: float = 0.2) -> tuple:
    "divide ASTLoader into trainset and devset"
    trainloader = DataLoader()
    devloader = DataLoader()
    if devname == None:
        pass
    else:
        for ast in astloader:
            if devname in ast.name:
                devloader.append(ast)
            else:
                trainloader.append(ast)
    return (trainloader, devloader)