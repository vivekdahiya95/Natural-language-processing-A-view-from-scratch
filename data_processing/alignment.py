"""
Given a sentence in hindi say example below, this functions takes this sentence and return the same sentence with
the following changes done to it:
1 remove all the leading and trailing spaces
2 if there is no space after . or ,  or ? then add a space after it
3 if there are greater than two continuous spaces then remove the extra spaces

this module takes a text file as argument name and then for each line in the file it does the above operations and
create a new file with the same name as the input file but with _cleaned appended to it

"""

import re
import argparse

def clean_hindi_sentence(sentence):
    sentence = re.sub(r"(?<=[।,?])", " ", sentence)
    sentence = re.sub(r" +", " ", sentence)
    sentence = sentence.strip()
    return sentence


if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="process the input file and return the clean file")
    parser.add_argument("input_file",type=str, help="input file to be processed")
    args = parser.parse_args()
    input_file_path = args.input_file
    with open(input_file_path, "r") as f:
        data = f.readlines()
    data = [clean_hindi_sentence(line) for line in data]
    with open(input_file_path+"_cleaned", "w") as f:
        f.writelines(data)



