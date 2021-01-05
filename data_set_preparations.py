import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_data_set(number_of_data_set):
    """
    Prepare the data set for the clustering.
    :param number_of_data_set: number of data set to be prepared
    :return: prepared data
    """
    if number_of_data_set == 1:
        return prepare_data_set1()
    elif number_of_data_set == 2:
        return prepare_data_set2()
    else:
        return prepare_data_set3()


def prepare_data_set1():
    """
    Prepare the first data set to clustering.
    :return: prepared data
    """
    data = pd.read_csv("dataset/online_shoppers_intention.csv")
    # drop 3 last columns
    data = data.drop(columns=['VisitorType', 'Weekend', 'Revenue'])
    print(data.columns)
    print(data.info)
    # months strings to int
    data['Month'] = data['Month'].astype('category')
    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    return data


def prepare_data_set2():
    """
    Prepare the second data set to clustering.
    :return: prepared data
    """
    data = pd.read_csv("dataset/diabetic_data.csv", skiprows=lambda x: x % 4 != 0)
    # drop race and gender
    data = data.drop(columns=['race', 'gender'])
    data = data.replace({'?': None})
    print(data)
    print(data.dtypes)
    for column in data.columns:
        if data.dtypes[column] == 'object':
            data[column] = data[column].astype('category')
    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    print(data.dtypes)
    print(data)
    print("Impute missing values with the median value and check again for missing values:")
    # impute with median
    for column in data.columns:
        data.loc[data[column].isnull(), column] = data[column].median()
    print(data.isna().sum())
    print("There are no missing values now")
    return data


def prepare_data_set3():
    """
    Prepare the third data set to clustering.
    :return: prepared data
    """
    data = pd.read_csv("dataset/e-shop clothing 2008.csv", sep=';', skiprows=lambda x: x % 10 != 0)
    data = data.drop(columns=['country'])
    data['page 2 (clothing model)'] = data['page 2 (clothing model)'].astype('category')
    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    print(data.columns)
    print(data.dtypes)
    print(data['page 2 (clothing model)'])
    return data


def scale_the_data(data):
    """
    Scales the data
    :param data: data to scale
    :return: scaled data
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)


if __name__ == '__main__':
    prepare_data_set1()
