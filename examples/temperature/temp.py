import pandas as pd


def read_data(option='daily'):
    '''Reads in a dataset with daily recorded temperatures for major cities of
       the world see
       https://www.kaggle.com/sudalairajkumar/daily-temperature-of-major-cities

        Returns:
            df: dataframe with columns region, country, city, date
                and temperature
    '''
    index_names = ['region', 'country', 'city', 'date']
    df = (pd.read_csv('city_temperature.csv', na_values=[-99],
                      low_memory=False)
            .rename(str.lower, axis='columns')
            .loc[lambda x: (x.day != 0) & (x.year != 200) & (x.year != 201)]
            .drop(['state'], axis=1)  # day 0, year; 200, 201 seem errors
            .assign(region=lambda x: x.region.astype('category'),
                    country=lambda x: x.country.astype('category'),
                    city=lambda x: x.city.astype('category'))
            .assign(avgtemperature=(
                    lambda x: x.avgtemperature.fillna(method='ffill')))
            .assign(temperature=lambda x: (x.avgtemperature-32)*5/9)
            .drop(['avgtemperature'], axis=1)  # Fahrenheit to Celcius
            .assign(date=lambda x: pd.to_datetime(x[['year', 'month', 'day']],
                    errors='coerce'))
            .drop(['year', 'month', 'day'], axis=1)
            .drop_duplicates(subset=index_names)  # NOTE: ideally take mean
            .dropna()
            .set_index(index_names)
            .sort_index(level=index_names)
          )
    if not option:
        options = ['monthly', 'biweekly', 'daily']
        print("Please choose:")
        for idx, element in enumerate(options):
            print("{}) {}".format(idx, element))
        choice = options[int(input("Enter number: "))]
        if choice == 'monthly':
            df = (df.groupby(['city', pd.Grouper(level='date', freq='m')])
                    .mean()
                  )
        elif choice == 'biweekly':
            df = (df.groupby(['city', pd.Grouper(level='date', freq='2W')])
                    .mean()
                  )
    return df
