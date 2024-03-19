import typing
class CreedsOptions():
    def __init__(self, options : map):

        standard = {
            "directory" : "../input/ligands.sdf",
            "parallel" : 1,
            
        }
        def getObject(options: map, key):
            value = options[key]
            if value == Null:
                return standard[key]

    