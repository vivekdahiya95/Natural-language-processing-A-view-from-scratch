"""
given two classes we have to sample the data from class 1 more times than class 2 since we have lesser samples in class1
"""


def balance_data(df, class_col, class1, class2):
    """
    balance data in the dataframe
    """
    df_class_1 = df[df[class_col] == class1]
    df_class_2 = df[df[class_col] == class2]
    df_class_1 = df_class_1.sample(df_class_2.shape[0])
    df = pd.concat([df_class_1, df_class_2])
    return df
