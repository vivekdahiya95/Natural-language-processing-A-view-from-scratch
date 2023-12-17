"""
write a code to analyze the hindi data
in this we are going to read a text file line by line and then we are going to find the following things
how many of these punctuations are there in the each line of the text file , ? |
how many of these words are there in each line of the text file
then we have to do basic analysis such as how many words are there in each line and how many punctuations are there in
each line average number of words and average number of punctuations in the whole data
plot this average data overall in the form of bar graph
then we need to write an algorithm to only select those lines which has more than one punctuation in each line only

"""


import codecs
import pandas as pd
import statistics
import matplotlib.pyplot as plt


def read_file(file_name):
    with codecs.open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        return lines


punctuations = [",", "|", "?"]


def get_punctuation_count(line):
    count = 0
    for char in line:
        if char in punctuations:
            count += 1
    return count


def get_line_length(line):
    return len(line.split())


def get_punctuation_count_list(lines):
    return [get_punctuation_count(line) for line in lines]


def get_line_length_list(lines):
    return [get_line_length(line) for line in lines]


def draw_histogram(x):
    pd.value_counts(x).plot.bar()
    # save the bar graph here as a png file
    plt.savefig("hindi_analysis.png")


def get_statistics(
    data, threshold=2
):  # threshold is the number of punctuations in each line
    count_greater_than_threshold = sum(1 for num in data if num >= threshold)
    # average of all numbers
    average = statistics.mean(data)
    return count_greater_than_threshold, average


# create a new function that gets the line and then analyse the line and if the line contains more than two punctuations
# then add that file to a list provided as a parameter to the function or described as a global variable
# then return that list


def get_lines_with_more_than_two_punctuations(lines):
    lines_with_more_than_two_punctuations = []
    for line in lines:
        if get_punctuation_count(line) >= 2:
            lines_with_more_than_two_punctuations.append(line)
    return lines_with_more_than_two_punctuations


# in the lines with more than two punctuations
# now create two separate list one that contains ? in them and other that contains | in them
# the input to the function is the list of lines with more than two punctuations
# the output of the function is two lists one that contains ? in them and other that contains | in them
# use regex to find the lines with ? and | in them
# then return the two lists


def get_lines_with_question_mark(lines):
    lines_with_question_mark = []
    for line in lines:
        if "?" in line:
            lines_with_question_mark.append(line)
    return lines_with_question_mark


def get_lines_with_pipe_and_no_question_mark(lines):
    # these lines shouldn't have question mark in them
    lines_with_pipe_and_no_question_mark = []
    for line in lines:
        if "|" in line and "?" not in line:
            lines_with_pipe_and_no_question_mark.append(line)


def main():
    lines = read_file("./hindi.txt")
    punctuation_counts_list = get_punctuation_count_list(lines)
    line_length_list = get_line_length_list(lines)
    # get the list with more than two punctuations
    lines_with_more_than_two_punctuations = get_lines_with_more_than_two_punctuations(
        lines
    )
    # get the list with question mark in them
    lines_with_question_mark = get_lines_with_question_mark(
        lines_with_more_than_two_punctuations
    )
    # get the list with pipe in them and no question mark in them
    lines_with_pipe_and_no_question_mark = get_lines_with_pipe_and_no_question_mark(
        lines_with_more_than_two_punctuations
    )
    # now save the above two lists in two separate files in the form of text files
    with open("lines_with_question_mark.txt", "w") as f:
        for line in lines_with_question_mark:
            f.write(line)
    with open("lines_with_pipe_and_no_question_mark.txt", "w") as f:
        for line in lines_with_pipe_and_no_question_mark:
            f.write(line)
    print(get_statistics(punctuation_counts_list, 2))
    draw_histogram(punctuation_counts_list)
    print(get_statistics(line_length_list, 2))
    draw_histogram(line_length_list)


if __name__ == "__main__":
    main()
