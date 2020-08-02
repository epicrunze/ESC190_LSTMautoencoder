import numpy as np
from fuzzywuzzy import process
import os

class Word:
    def __init__(self, word: str, definition: list, encoded_vec: list, sort_fac = "word"):
        self.word = word
        self.definition = definition
        self.vec = encoded_vec
        self.sort_fac = sort_fac

    def eval(self):
        if self.sort_fac == "word":
            return self.word
        elif self.sort_fac == "vec":
            return self.vec[0]

class Node:
    def __init__(self, word: Word, left = None, right= None, parent= None, height: int = 1, bf: int = 0):
        self.word = word
        self.left = left
        self.right = right
        self.parent = parent
        self.height = height
        self.bf = bf
    
    def __gt__(self, other):
        if isinstance(other, Node):
            return self.word.eval() > other.word.eval()
        return self.word.eval() > other
    def __lt__(self, other):
        if isinstance(other, Node):
            return self.word.eval() < other.word.eval()
        return self.word.eval() < other
    def __ge__(self, other):
        if isinstance(other, Node):
            return self.word.eval() >= other.word.eval()
        return self.word.eval() >= other
    def __le__(self, other):
        if isinstance(other, Node):
            return self.word.eval() <= other.word.eval()
        return self.word.eval() <= other
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.word.eval() == other.word.eval()
        return self.word.eval() == other
    def __ne__(self, other):
        if isinstance(other, Node):
            return self.word.eval() != other.word.eval()
        return self.word.eval() != other
      
class Dictionary:
    def __init__(self, root_node: Node = None):
        self.root = root_node
        self.dist_thresh = 80
        self.VEC_SIZE = 512
        self.encoding_model = None
        self.generating_model = None
        self.punc_list = None
        self.char_list = None
        self.word2vec = None

    def balanced_insert(self, node, curr = None):
        if self.root == None:
            self.root = node
            return
        curr = curr if curr else self.root
        self.insert(node, curr)
        self.balance_tree(node)

    def search_helper(self, val: str, node: Node):
        if node:
            if node == val:
                return node.word
            elif node > val:
                return self.search_helper(val, node.left)
            else:
                return self.search_helper(val, node.right)
        return None

    def search(self, val:str):
        return self.search_helper(val, self.root)

    def subset_search(self, val: str, node = "yeet"):
        node = node if node != "yeet" else self.root
        if node:
            if node == val:
                return node
            elif node > val:
                return self.subset_search(val, node.left)
            else:
                return self.subset_search(val, node.right)
        return None

    def search_substrings(self, val: str, node = None):
        curr = node if node else self.root
        returnlist = []
        lookupqueue = [curr]
        while lookupqueue:
            curr = lookupqueue.pop()
            if curr:
                if curr == val:
                    returnlist.append(curr.word.eval())
                    lookupqueue.append(curr.right)
                elif curr > val:
                    if curr.word.eval().startswith(val):
                        returnlist.append(curr.word.eval())
                        lookupqueue.append(curr.right)
                    lookupqueue.append(curr.left)
                elif curr < val:
                    if curr.word.eval().startswith(val):
                        returnlist.append(curr.word.eval())
                        lookupqueue.append(curr.left)
                    lookupqueue.append(curr.right)
        return returnlist

    def insert(self, node: Node, curr: Node = None):
        if self.root == None:
            self.root = node
            return
        curr = curr if curr else self.root
        # insert at correct location in BST
        if node < curr:
            if curr.left is not None:
                self.insert(node, curr.left)
            else:
                node.parent = curr
                curr.left = node
        else:
            if curr.right is not None:
                self.insert(node, curr.right)
            else:
                node.parent = curr
                curr.right = node
        return

    def delete(self, root, key):
        if not root:
            return root
        elif key < root:
            root.left = self.delete(root.left, key)
        elif key > root:
            root.right = self.delete(root.right, key)
        else:
            if root.left is None:
                temp = root.right
                root = None
                return temp
            elif root.right is None:
                temp = root.left
                root = None
                return temp
            temp = self.getMinValueNode(root.right)
            root.word = temp.word
            root.right = self.delete(root.right, temp.word.eval())

        #If the tree has a single node return node
        if root is None:
            return root
        #update the height of the tree
        self.update_height(root)
        #find the balance factor
        root.bf = self.find_balance_factor(root)
        #Left Left
        if root.bf > 1 and self.find_balance_factor(root.left) >= 0:
            return self.left_rotate(root)
        #Right Right
        if root.bf < -1 and self.find_balance_factor(root.right) <= 0:
            return self.right_rotate(root)
        #Left Right
        if root.bf > 1 and self.find_balance_factor(root.left) < 0:
            root.left = self.right_rotate(root.left)
            return self.left_rotate(root)
        #Right Left
        if root.bf < -1 and self.find_balance_factor(root.right) < 0:
            root.right = self.left_rotate(root.right)
            return self.right_rotate(root)
        return root

    def getMinValueNode(self, curr): 
        if curr is None or curr.left is None: 
            return curr
        return self.getMinValueNode(curr.left) 

    def balance_tree(self, node: Node):
        head = node
        while not self.is_balanced():
            new_head = head.parent
            if head.bf == 1:
                if head.parent.bf == -2:
                    self.left_rotate(head)
                    head = new_head
                    continue
            elif head.bf > 1:
                self.left_rotate(head)
                head = new_head
                continue
            elif head.bf == -1:
                if head.parent.bf == 2:
                    self.right_rotate(head)
                    head = new_head
                    continue
            elif head.bf < -1:
                self.right_rotate(head)
                head = new_head
                continue
            head = new_head

    def update_height(self, node):
        node.height = 1 + max(self.height(node.left), self.height(node.right))

    def height(self, node):
        return node.height if node else -1

    def left_rotate(self, z):
        y = z.right
        y.parent = z.parent
        if y.parent is None:
            self.root = y
        else:
            if y.parent.left is z:
                y.parent.left = y
            elif y.parent.right is z:
                y.parent.right = y
        z.right = y.left
        if z.right is not None:
            z.right.parent = z
        y.left = z
        z.parent = y
        self.update_height(z)
        self.update_height(y)
        return y

    def right_rotate(self, z):
        new_root = z.left
        new_root.parent = z.parent
        if not new_root.parent:
            self.root = new_root
        else:
            if new_root.parent.left is z:
                new_root.parent.left = new_root
            elif new_root.parent.right is z:
                new_root.parent.right = new_root
        z.left = new_root.right
        if z.left:
            z.left.parent = z
        new_root.right = z
        z.parent = new_root
        self.update_height(z)
        self.update_height(new_root)
        return new_root

    def find_balance_factor(self, node):
        if not node:
            return 0
        r_height = -1
        l_height = -1
        if node.right:
            r_height = node.right.height
        if node.left:
            l_height = node.left.height
        return r_height - l_height

    def post_order_balancing(self, node, max_bf):
        if node:
            self.post_order_balancing(node.left, max_bf)
            self.post_order_balancing(node.right, max_bf)
            self.update_height(node)
            node.bf = self.find_balance_factor(node)
            if abs(node.bf) > max_bf[0]:
                max_bf[0] = abs(node.bf)

    def is_balanced(self):
        max_bf = [0]
        self.post_order_balancing(self.root, max_bf)
        if max_bf[0] > 1:
            return False
        else:
            return True

    def return_dict(self, curr="yeet"):
        curr = curr if curr != "yeet" else self.root
        listy = []
        if curr:
            listy.extend(self.return_dict(curr.left))
            listy.append(curr.word.word)
            listy.extend(self.return_dict(curr.right))
        return listy

    def __str__(self):
        return str(self.return_dict())

    def write_helper(self, file, curr="yeet"):
        curr = curr if curr != "yeet" else self.root
        if curr:
            self.write_helper(file, curr.left)
            file.write("{word} {defs}\n".format(word=curr.word.word, defs="; ".join(curr.word.definition)))
            self.write_helper(file, curr.right)

    def write2file(self, filename):
        with open(filename, "w") as file:
            self.write_helper(file)
        return

    def findWords(self):
        word = input("Enter a word:\n")
        word = self.search(word)
        if word:
            print("for the word: {}".format(word.word))
            for definition in word.definition:
                print("Definition: {}".format(definition))
        else:
            print("no word found")
        input()

    def insertWord(self):
        word = input("Please enter a word that you want to put into the dictionary:\n")
        definition = input("Please enter some definitions, seperated by semicolons:\n").split(";")
        if word and definition: 
            encoded_vec = definition
            self.insert(Node(Word(word, definition, encoded_vec)))
    
    def deleteWord(self):
        word = input("Please input a word that you would like to delete:\n")
        if word:
            deletedWord = self.delete(self.root, word)
            if deletedWord:
                print("Successfully deleted: ", deletedWord.word.word)
            else:
                print("Word: {} not deleted, please try again.".format(word))
        else:
            print("invalid input")

    def scrollWord(self):
        word = input("Please input a word:\n")
        if word and self.search(word):
            word_list = self.return_dict()
            index = word_list.index(word)
            for each in word_list[index-4:index+4]:
                print(each)
            while True:
                print("scroll up with: u, scroll down with d, exit with q")
                command = input()
                os.system('cls')
                if command == "u":
                    index -= 3
                if command == "d":
                    index += 3
                if command == "q":
                    return
                try:
                    for each in word_list[index-4:index+4]:
                        print(each)
                except:
                    if command == "u":
                        index += 3
                    if command == "d":
                        index -= 3
    
    def vec2predictions(self, inputvec, word_arr, vec_arr, temperature = 0.005, prob = True, num_samples = 1):
        import numpy as np
        import tensorflow as tf
        import numpy.linalg as LA
        '''
        inputvec => np.array of shape (variable, VECTOR_SIZE)
        word_arr => np.array of shape (vocab_size, 1)
        vec_arr => np.array of shape (vocab_size, VECTOR_SIZE)
        returns a np.array of shape (variable), with words
        '''
        dotted_prod = inputvec @ vec_arr.T
        norm_vec_arr = LA.norm(vec_arr, axis=1)
        norm_inputvec = LA.norm(inputvec, axis=1)
        divisor = np.expand_dims(norm_inputvec.T, 1) @ np.expand_dims(norm_vec_arr, 0)
        
        finalvec = dotted_prod / divisor
        finalvec = finalvec / temperature
        if prob:
            indicies = tf.random.categorical(finalvec, num_samples=num_samples)
            indicies = np.squeeze(indicies)
        else:
            indicies = np.argsort(-finalvec, axis=-1)[:, :num_samples]
            indicies = np.squeeze(indicies)

        #print("computed final prediction vector: with shape", indicies.shape)

        output = word_arr[indicies]
        return output

    def processString(self, inputstr: str, punc_list, charlist, word2vec):
        '''
        returns a list of fully processed strings with guarenteed mappings with alpha num chars and punctuation
        '''
        outputlist = []
        for char in inputstr:
            char = char.lower()
            if char in punc_list or char in charlist or char == " ":
                outputlist.append(char)

        outputstr = "".join(outputlist)

        for each in punc_list:
            outputstr = outputstr.replace(each, " {} ".format(each))

        outputstr = outputstr.split(" ")
        outputstr = " ".join(outputstr)

        outputlist = []

        for each in outputstr.split(" "):
            if each in word2vec:
                outputlist.append(each)
        
        return outputlist

    def generateText(self, model, start_string, word2vec, punc_list, char_list, num_generate = 15, temperature = 0.005):
        import numpy as np
        import tensorflow as tf
        word_arr = []
        vec_arr = []
        for word, vector in word2vec.items():
            word_arr.append(word)
            vec_arr.append(vector)
        word_arr = np.array(word_arr)
        vec_arr = np.array(vec_arr)

        inputvec = []
        for each in self.processString(start_string, punc_list, char_list, word2vec):
            inputvec.append(word2vec[each])

        inputvec = np.array(inputvec)

        inputvec = tf.expand_dims(inputvec, 0)

        outputText = []
        model.reset_states()
        for i in range(num_generate):
            #print("starting prediction")
            pred = model.predict(inputvec)
            #print("finished prediction: output shape = ", pred.shape)
            # divide prediction by temperature to do stuff
            pred = tf.expand_dims(pred[:, -1, :], 0)
            pred = tf.squeeze(pred, 0)
            predicted_word = self.vec2predictions(pred, word_arr, vec_arr, temperature)
            #print("predicted word is: ", predicted_word)
            outputText.append(predicted_word)

            inputvec = tf.expand_dims(tf.expand_dims(word2vec[predicted_word], 0), 0)

        return (start_string + " ".join(outputText))

    def run(self):
        while True:
            print("Dictionary: created by Ryan Zhang")
            print("1: find definition to input word")
            print("2: Insert a given word and definition")
            print("3: Delete word")
            print("4: Scroll around a given word")
            print("5: Find synonyms")
            print("6: Find closest word")
            print("7: Find words that follow from a segment")
            print("8: Size of dictionary")
            print("9: Write dictionary to a file")
            print("10: Generate story")
            print("type: quit to quit")
            mode = input("Enter a mode:\n")
            os.system('cls')
            if mode == "1":
                self.findWords()
            elif mode == "2":
                self.insertWord()
            elif mode == "3":
                self.deleteWord()
            elif mode == "4":
                self.scrollWord()
            elif mode == "5":
                if self.word2vec:
                    import numpy as np
                    word_arr = []
                    vec_arr = []
                    for word, vector in self.word2vec.items():
                        word_arr.append(word)
                        vec_arr.append(vector)
                    word_arr = np.array(word_arr)
                    vec_arr = np.array(vec_arr)

                    inpstr = input("Please input a word:\n")
                    if inpstr in self.word2vec:
                        inputvec = np.expand_dims(self.word2vec[inpstr], 0)
                    else:
                        print("not found")
                        continue 
                    num_samples = int(input("how many related words? \n"))
                    print("Finding related words")
                    for each in self.vec2predictions(inputvec, word_arr, vec_arr, prob = False, num_samples = num_samples):
                        if each != inpstr:
                            print(each)
                    input()
                else:
                    import pickle
                    inpstr = input("provide the pickle file of the word2vec:\n")
                    self.word2vec = pickle.load(open(inpstr, "rb"))
            elif mode == "6":
                word = input("Please enter a word:\n")
                if word:
                    output = process.extractOne(word, self.return_dict())
                    if output:
                        if output[1] > self.dist_thresh:
                            print("The closest word found is:\n", output[0])
            elif mode == "7":
                substr = input("Please enter a substring:\n")
                for each in self.search_substrings(substr):
                    print(each)
                input()
            elif mode == "8":
                print("The size of the dictionary is: {}".format(len(self.return_dict())))
                input()
            elif mode == "9":
                filename = input("input a file name:\n")
                if filename:
                    self.write2file(filename)
            elif mode == "10":
                if self.generating_model:
                    inpstr = input("Starting string:\n")
                    num = int(input("number of words to generate:\n"))
                    temp = float(input("Temperature, default 0.01, lower the temp, the more predictable:\n"))
                    print(self.generateText(model=self.generating_model, start_string=inpstr, word2vec=self.word2vec, 
                    punc_list=self.punc_list, char_list=self.char_list, num_generate=num, temperature=temp))
                else:
                    print("please load in a model")
                pass
            elif mode == "quit":
                return
            else:
                print("Invalid input, try again")
            
