import pandas as pd
import warnings
import numpy as np
from matplotlib import pyplot as plt

warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def extract_data_before_2019y():
    """
    Extracts data from the 302-19.xlsx file
    :return: pandas dataframe with columns 'Дата', 'Задолженность', 'Просроченная задолженность'
    """
    return pd.read_excel('302-19.xlsx', usecols=[0, 5, 11], skiprows=list(range(7)),
                          names=['Дата', 'Задолженность', 'Просроченная задолженность'])


def extract_data_after_2018():
    """
    Extracts data from the 01_13_F_Debt_sme_subj.xlsx file
    :return: pandas dataframe with columns 'Дата', 'Задолженность', 'Просроченная задолженность'
    """
    # read Задолженность from the page МСП Итого
    # .T to make rows for entities and columns for properties
    after_19y_debt = pd.read_excel('01_13_F_Debt_sme_subj.xlsx', skiprows=1, nrows=1, sheet_name='МСП Итого ').T
    after_19y_debt.reset_index(inplace=True)
    # remove odd row after transpose
    after_19y_debt.drop(labels=0, axis=0, inplace=True)
    after_19y_debt.columns = before_19y.columns[:2]

    # change types of columns for convenience
    after_19y_debt[after_19y_debt.columns[0]] = pd.to_datetime(after_19y_debt[after_19y_debt.columns[0]])
    after_19y_debt = after_19y_debt.astype({after_19y_debt.columns[1]: 'int32'}, copy=False)

    # read Просроченная задолженность from the page МСП в т.ч. просроч.
    after_19y_prosro4eno = pd.read_excel('01_13_F_Debt_sme_subj.xlsx', skiprows=2, nrows=0,
                                         sheet_name='МСП в т.ч. просроч.').T
    after_19y_prosro4eno.reset_index(inplace=True)
    # remove odd row after transpose
    after_19y_prosro4eno.drop(labels=0, axis=0, inplace=True)
    # name column
    after_19y_prosro4eno.columns = ['Просроченная задолженность']

    # concatenate Задолженность and Просроченная задолженность in one table and return it
    return pd.concat([after_19y_debt, after_19y_prosro4eno], axis=1)


def transform_to_quarters_format(debt_table):
    """
    Transforms debt_table from month format to quarters taking average for each quarter
    :param debt_table:
    :return: table in correct quarter format with averaged values in columns
    """
    debt_table_quarters = pd.DataFrame()

    # creates array [1, 1, 1, 2, 2, 2, 3, 3, 3, ...], so i-th month will be from corresponding quarter
    correct_quarters = np.ones((debt_table.shape[0] // 3 + 3, 3), dtype=int).cumsum(axis=0).flatten()
    # quarter of the first month in the data
    first_quarter = (debt_table['Дата'].dt.month[0] - 1) // 3 + 1
    # assumption: data is not missing a single month
    # then quarters are from correct_quarters continuous part from [first_quarter to first_quarter + number of months]
    debt_table['Квартал'] = correct_quarters[first_quarter: debt_table.shape[0] + first_quarter]

    # calculate average value inside each quarter and assign those values to the resulting table
    group = debt_table.groupby('Квартал')
    debt_table_quarters_0 = group['Задолженность'].mean()  # avg for Задолженность inside quarters
    debt_table_quarters_1 = group['Просроченная задолженность'].mean()  # for Просроченная задолженность
    debt_table_quarters = pd.concat([debt_table_quarters_0, debt_table_quarters_1], axis=1)
    debt_table_quarters.reset_index(inplace=True)

    return debt_table_quarters


if __name__ == '__main__':
    before_19y = extract_data_before_2019y()
    after_19y = extract_data_after_2018()

    # concatenates old and new data
    debt_table_total = pd.concat([before_19y, after_19y])
    debt_table_total.reset_index(inplace=True)
    debt_table_total.drop('index', 1, inplace=True)

    debt_table_quarters_format = transform_to_quarters_format(debt_table_total)

    # plot data before quarters averaging
    debt_table_total.plot(x='Дата', y=['Задолженность', 'Просроченная задолженность'])
    plt.show()

    # ... and after
    debt_table_quarters_format.plot(x=['Квартал', 'Квартал'], y=['Задолженность', 'Просроченная задолженность'],
                                    kind='scatter')
    plt.show()
