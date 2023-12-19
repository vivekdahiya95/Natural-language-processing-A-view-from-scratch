import matplotlib.pyplot as plt
import numpy as np
import string
import codecs
import unicodedata
import pandas as pd
from collections import Counter
import statistics


def read_file(file_name):
    with codecs.open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        return lines


def get_punctuation_count(line):
    count = 0
    for char in line:
        if char in string.punctuation:
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
    plt.show()


def get_statistics(data, threshold=3):
    count_greater_than_threshold = sum(1 for num in data if num >= threshold)

    ##average of all numbers
    average = statistics.mean(data)

    return count_greater_than_threshold, average


def main():
    lines = read_file("./viet.txt")
    punctuation_counts_list = get_punctuation_count_list(lines)
    line_length_list = get_line_length_list(lines)
    # x=Counter(punctuation_counts_list)
    # print(x)
    print(get_statistics(punctuation_counts_list, 3))
    draw_histogram(punctuation_counts_list)


if __name__ == "__main__":
    main()
