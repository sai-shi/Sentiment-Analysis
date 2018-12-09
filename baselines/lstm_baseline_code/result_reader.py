# result_reader.py
# MES 3/15/18
import json
#
infile = 'result/lstm.txt'
# infile = 'result-lstm-clean-no-spellck/lstm.txt'
# infile = 'result-lstm-processed/lstm.txt'
with open(infile, 'r') as f:
    details = json.load(f)
# with open(infile, 'r') as f:
#     details = json.loads(f.readlines())
#
summary = []
for key, value in list(details.items()):
    keyparts = key.split('_')
    if keyparts[0] == 'acc':
        # treat as dictionary
        print(" ")
        print(key)
        print("epoch   binary-result   three-way result")
        i = 0
        for v_item in value:
            print_string = "%s: %3d     %s   %s" % (key, i, v_item['binary'], v_item['three-way'])
            print(print_string)
            if i == 24:
                summary.append(print_string)
            i += 1
    else:
        # treat as list
        print(" ")
        print(key)
        print("epoch   result")
        i = 0
        for v_item in value:
            print("%3d     %s" % (i, str(v_item)))
            i += 1
    #
    # print("%s = %s" % (key, str(value)))
print(" ")
print("Summary:")
print("           epoch   binary-result   three-way result")
for item in summary:
    print(item)
