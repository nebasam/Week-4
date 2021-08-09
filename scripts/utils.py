


supported = """a b c d 
e f g h
i j k l
m n o p
q r s t
u v w x 
y z
""".split()

char_map = {}
char_map[""] = 0
char_map["<SPACE>"] = 1
index = 2
for c in supported:
    char_map[c] = index
    index += 1
index_map = {v+1: k for k, v in char_map.items()}



def text_to_int_sequence(text):
    """ Convert text to an integer sequence """
    print("reliad")
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            # print("checking character " + c + " in map:")
            # print(char_map)
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence



def int_sequence_to_text(int_sequence):
    """ Convert an integer sequence to text """
    text = []
    for c in int_sequence:
        if(c==0):
            pass
        else:
            ch = index_map[c]
            text.append(ch)
    return text