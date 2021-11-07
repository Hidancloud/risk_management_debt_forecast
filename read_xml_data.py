import pandas as pd
import warnings
import numpy as np
from matplotlib import pyplot as plt

warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

online = True  # if True: download xml files from github URL
# be careful: online version will not work if requirements from requirements.txt are not satisfied!

if online:
    url_link_302_19 = 'https://github.com/Hidancloud/risk_management_debt_forecast/' \
                      'blob/main/data_folder/302-19.xlsx?raw=true'

    url_link_01_13_F_Debt_sme_subj = 'https://github.com/Hidancloud/risk_management_debt_forecast/' \
                                     'blob/main/data_folder/01_13_F_Debt_sme_subj.xlsx?raw=true'

    url_link_Interpolationexp2 = 'https://github.com/Hidancloud/risk_management_debt_forecast/' \
                                 'blob/main/data_folder/Interpolationexp2.xlsx?raw=true'


def extract_data_before_2019y():
    """
    Extracts data from the 302-19.xlsx file
    :return: pandas dataframe with columns 'Дата', 'Задолженность', 'Просроченная задолженность'
    """
    if online:
        return pd.read_excel(url_link_302_19, usecols=[0, 5, 11], skiprows=list(range(7)),
                             names=['Дата', 'Задолженность', 'Просроченная задолженность'])

    return pd.read_excel('data_folder/302-19.xlsx', usecols=[0, 5, 11], skiprows=list(range(7)),
                         names=['Дата', 'Задолженность', 'Просроченная задолженность'])


def extract_data_after_2018():
    """
    Extracts data from the 01_13_F_Debt_sme_subj.xlsx file
    :return: pandas dataframe with columns 'Дата', 'Задолженность', 'Просроченная задолженность'
    """
    # read Задолженность from the page МСП Итого
    # .T to make rows for entities and columns for properties
    if online:
        after_19y_debt = pd.read_excel(url_link_01_13_F_Debt_sme_subj, skiprows=1, nrows=1,
                                       sheet_name='МСП Итого ').T
    else:
        after_19y_debt = pd.read_excel('data_folder/01_13_F_Debt_sme_subj.xlsx',
                                       skiprows=1, nrows=1, sheet_name='МСП Итого ').T

    after_19y_debt.reset_index(inplace=True)
    # remove an odd row after transpose
    after_19y_debt.drop(labels=0, axis=0, inplace=True)
    after_19y_debt.columns = before_19y.columns[:2]

    # change types of the columns for convenience
    after_19y_debt[after_19y_debt.columns[0]] = pd.to_datetime(after_19y_debt[after_19y_debt.columns[0]])
    after_19y_debt = after_19y_debt.astype({after_19y_debt.columns[1]: 'int32'}, copy=False)

    # read Просроченная задолженность from the page МСП в т.ч. просроч.
    if online:
        after_19y_prosro4eno = pd.read_excel(url_link_01_13_F_Debt_sme_subj, skiprows=2, nrows=0,
                                             sheet_name='МСП в т.ч. просроч.').T
    else:
        after_19y_prosro4eno = pd.read_excel('data_folder/01_13_F_Debt_sme_subj.xlsx', skiprows=2, nrows=0,
                                             sheet_name='МСП в т.ч. просроч.').T
    after_19y_prosro4eno.reset_index(inplace=True)
    # remove an odd row after the transpose
    after_19y_prosro4eno.drop(labels=0, axis=0, inplace=True)
    # name the column
    after_19y_prosro4eno.columns = ['Просроченная задолженность']

    # concatenate Задолженность and Просроченная задолженность in one table and return it
    return pd.concat([after_19y_debt, after_19y_prosro4eno], axis=1)


def extract_macro_parameters():
    if online:
        return pd.read_excel(url_link_Interpolationexp2, index_col=0, parse_dates=True)
    return pd.read_excel('data_folder/Interpolationexp2.xlsx', index_col=0, parse_dates=True)


def transform_to_quarters_format(custom_table, date_column_name='Дата', already_3month_correct_step=False):
    """
    Transforms table from month format to quarters taking average for each quarter if necessary
    :param custom_table: Pandas dataframe
    :param date_column_name: name of a column with dates
    :param if the time step between custom_table rows is a 3 month instead of month and correspond to 3, 6, 9, 12 months
    :return: table in correct quarter format with averaged values in columns
    """
    if not already_3month_correct_step:
        # creates array [1, 1, 1, 2, 2, 2, 3, 3, 3, ...], so i-th month will be from corresponding quarter
        # in case when each row corresponds to a month
        correct_quarters = np.ones((custom_table.shape[0] // 3 + 3, 3), dtype=int).cumsum(axis=0).flatten()
        # quarter of the first month in the data
        first_quarter = (custom_table[date_column_name].dt.month[0] - 1) // 3 + 1
        # assumption: the data is not missing a single month
        # then quarters are from correct_quarters continuous part: [first_quarter to first_quarter + number of months]
        custom_table['Квартал'] = correct_quarters[first_quarter: custom_table.shape[0] + first_quarter]
    else:
        # in case when each row corresponds to either 3, 6, 9 or 12 month (file with macro data)
        debt_table_quarters = custom_table.copy()
        debt_table_quarters.reset_index(inplace=True)
        debt_table_quarters['Квартал'] = custom_table.index.month // 3
        return debt_table_quarters

    # calculate average value inside each quarter and assign those values to the resulting table
    group = custom_table.groupby('Квартал')
    debt_table_quaters_features = dict()
    for feature in custom_table.columns:
        if feature != date_column_name and feature != 'Квартал':
            debt_table_quaters_features[feature] = group[feature].mean()
    debt_table_quarters = pd.concat(debt_table_quaters_features, axis=1)
    debt_table_quarters.reset_index(inplace=True)

    return debt_table_quarters


if __name__ == '__main__':
    # read the files
    before_19y = extract_data_before_2019y()
    after_19y = extract_data_after_2018()
    new_features = extract_macro_parameters()

    # concatenates old and new data
    debt_table_total = pd.concat([before_19y, after_19y])
    debt_table_total.reset_index(inplace=True)
    debt_table_total.drop('index', 1, inplace=True)

    debt_table_quarters_format = transform_to_quarters_format(debt_table_total, date_column_name='Дата')
    debt_table_quarters_format['Уровень просроченной задолженности'] = \
        debt_table_quarters_format['Просроченная задолженность'] / debt_table_quarters_format['Задолженность']

    # plot data before quarters averaging
    debt_table_total.plot(x='Дата', y=['Задолженность', 'Просроченная задолженность'])
    plt.show()

    # ... and after
    debt_table_quarters_format.plot(x=['Квартал', 'Квартал'], y=['Задолженность', 'Просроченная задолженность'],
                                    kind='scatter')
    plt.show()

    # add macro features:
    interpolated_new_features = new_features.interpolate(method='time', limit_direction='both', downcast='infer')
    interpolated_new_features_quarter_format = \
        transform_to_quarters_format(interpolated_new_features, date_column_name='Отчетная дата (по кварталам)',
                                     already_3month_correct_step=True)

    all_features = pd.concat([debt_table_quarters_format, interpolated_new_features_quarter_format], axis=1)
    all_features = all_features.iloc[:, :-1]  # removing an odd last column
    all_features.to_excel('Dataset.xlsx', index=False)  # save the dataset into the project directory
