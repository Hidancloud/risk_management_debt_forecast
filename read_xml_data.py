import pandas as pd
import warnings
from matplotlib import pyplot as plt

warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000


if __name__ == '__main__':

    before_19y = pd.read_excel('302-19.xlsx', usecols=[0, 5, 11], skiprows=list(range(7)),
                               names=['Дата', 'Задолженность', 'Просроченная задолженность'])

    after_19y_debt = pd.read_excel('01_13_F_Debt_sme_subj.xlsx', skiprows=1, nrows=1, sheet_name='МСП Итого ').T
    after_19y_debt.reset_index(inplace=True)
    after_19y_debt.drop(labels=0, axis=0, inplace=True)
    after_19y_debt.columns = before_19y.columns[:2]

    after_19y_debt[after_19y_debt.columns[0]] = pd.to_datetime(after_19y_debt[after_19y_debt.columns[0]])
    after_19y_debt = after_19y_debt.astype({after_19y_debt.columns[1]: 'int32'}, copy=False)
    #

    after_19y_prosro4eno = pd.read_excel('01_13_F_Debt_sme_subj.xlsx', skiprows=2, nrows=0, sheet_name='МСП в т.ч. просроч.').T
    after_19y_prosro4eno.reset_index(inplace=True)
    after_19y_prosro4eno.drop(labels=0, axis=0, inplace=True)

    after_19y_prosro4eno.columns = ['Просроченная задолженность']

    after_19y = pd.concat([after_19y_debt, after_19y_prosro4eno], axis=1)

    debt_table = pd.concat([before_19y, after_19y])
    debt_table.reset_index(inplace=True)
    debt_table.drop('index', 1, inplace=True)

    print(debt_table)
    debt_table.plot(x='Дата', y=['Задолженность', 'Просроченная задолженность'])
    plt.show()

    '''
    import holoviews as hv
    import hvplot.pandas
    
    pd.options.plotting.backend = 'hvplot'
    
    hv.extension('bokeh')
    plot = debt_table.hvplot.scatter(x='Дата', y=['Задолженность', 'Просроченная задолженность'])

    from bokeh.plotting import show

    show(hv.render(plot))
    '''