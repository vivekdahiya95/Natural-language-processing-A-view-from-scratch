"""
Given a sentence in hindi like example below, this function returns a list of smaller sentences which follow the following rules:
given sentence is
इस समय अयोध्या को लिखना मुश्किल हो रहा है, चारों तरफ एक बेचैनी सी है। इसके साथ ही एक अजीब किस्म की खुशी। सब कुछ थोड़ा तेज सा चलने लगा है। श्रद्धालुओं के मन में एक ही सवाल है पता नहीं उसको दर्शन होंगे या नहीं। पुलिस वालों को लग रहा है की कहीं कोई चूक न हो जाए। दुकानदार ग्राहकों को सही से संभाल नहीं पा रहे हैं। सफाईकर्मियों के हाथ तेजी से झाड़ू लगाने और कचरे को हटाने में लगे हैं। इन सब के बीच में पुलिस का सायरन और अधिकारियों की गाड़ियों की दौड़। पूरी दुनिया का मीडिया यहां है और सबलोग सबकुछ कवर करने को उतारू। बीच-बीच नांचती हुईं बाबाओं की टोलियां और दूर-दूर से आए हुए लोग। जिन्होंने कैमरा नहीं देखा वो उन्हें ऐसे देख रहे हैं जैसे ये किसी दूसरे देश का प्राणी है। कुछ लोग इस बात की कोशिश कर रहे हैं कि किसी तरह उन्हें कैमरे में आने का मौका मिल जाए।
output should follow these rules
1. smaller sentences should be of length 35
2. smaller sentences should not break a word in between
3. each smaller sentence should end either in ।  or ?
4. output should be a list
5. keep concatenating the smaller sentences till the maximum length of sentence is 35
6. if a sentence is of length in between 15-20 then don't concatenate it with the next sentence

"""
import re


def split_hindi_sentence(sentence):
    sentence = sentence.strip()
    # split the sentence at । or ? and return a list of sentences and do preserve the । or ? at the end of the sentence
    result = re.split("।", sentence)[:-1]
    result = [item + "।" for item in result]
    # for each element in the result list split the components at ?
    for i in range(len(result)):
        if "?" in result[i]:
            temp = result[i].split("?")
            temp = [item + "?" for item in temp if item[-1] != "।"]
            result[i] = temp
    ## flatten the list given a list of strings which contains inside a list of strings also
    result = [
        item
        for sublist in result
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]
    print(result)
    print("#"*100)
    return result


def concatenate_sentence(sentence_list, threshold=35):
    """
    given a list of splited sentences, concatenate them to form a sentence of length 35
    """
    result = []
    temp = ""
    for i in range(len(sentence_list)):
        if len(temp.split()) + len(sentence_list[i].split()) <= threshold:
            temp += sentence_list[i]
        else:
            result.append(temp)
            temp = ""
            temp += sentence_list[i]
    result.append(temp)
    return result


if __name__ == "__main__":
    sentence = "इस समय अयोध्या को लिखना मुश्किल हो रहा है? चारों तरफ एक बेचैनी सी है। इसके साथ ही एक अजीब किस्म की खुशी। सब कुछ थोड़ा तेज सा चलने लगा है। श्रद्धालुओं के मन में एक ही सवाल है पता नहीं उसको दर्शन होंगे या नहीं। पुलिस वालों को लग रहा है की कहीं कोई चूक न हो जाए। दुकानदार ग्राहकों को सही से संभाल नहीं पा रहे हैं। सफाईकर्मियों के हाथ तेजी से झाड़ू लगाने और कचरे को हटाने में लगे हैं। इन सब के बीच में पुलिस का सायरन और अधिकारियों की गाड़ियों की दौड़। पूरी दुनिया का मीडिया यहां है और सबलोग सबकुछ कवर करने को उतारू। बीच-बीच नांचती हुईं बाबाओं की टोलियां और दूर-दूर से आए हुए लोग। जिन्होंने कैमरा नहीं देखा वो उन्हें ऐसे देख रहे हैं जैसे ये किसी दूसरे देश का प्राणी है। कुछ लोग इस बात की कोशिश कर रहे हैं कि किसी तरह उन्हें कैमरे में आने का मौका मिल जाए।"
    splitted_sentence = split_hindi_sentence(sentence)
    # given a list of splited sentences, concatenate them to form a sentence of length 35
    result = concatenate_sentence(splitted_sentence)
    print(result)
