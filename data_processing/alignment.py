"""
Given a sentence in hindi say example below, this functions takes this sentence and return the same sentence with
the following changes done to it:
1 remove all the leading and trailing spaces
2 if there is no space after . or ,  or ? then add a space after it
3 if there are greater than two continuous spaces then remove the extra spaces
"""

import re


def clean_hindi_sentence(sentence):
    sentence = re.sub(r"(?<=[।,?])", " ", sentence)
    sentence = re.sub(r" +", " ", sentence)
    sentence = sentence.strip()
    return sentence


if __name__ == "__main__":
    sentence = "इस समय अयोध्या को लिखना मुश्किल  हो रहा है?चारों तरफ एक बेचैनी सी है."
    print(clean_hindi_sentence(sentence))


