import structs


def process_line(string):
    '''takes in a line of the format: <word> <def>; <def>;...
    returns words: str, defs: list
    '''
    defs = []
    splitted = string.split()
    word = splitted.pop(0)
    for each in (" ".join(splitted)).split(";"):
        defs.append(each.strip())
    return word, defs

def data2dict(datafile):
    dictionary = None
    with open(datafile, "r") as input_doc:
        for line in input_doc:
            word, defs = process_line(line)
            #process defs
            embeddings = defs
            nodeyboi = structs.Node(structs.Word(word, defs, embeddings))
            if not dictionary:
                dictionary = structs.Dictionary(nodeyboi)
                continue
            dictionary.balanced_insert(nodeyboi)
    return dictionary