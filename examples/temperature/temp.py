import pandas as pd


def read_data():
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
            # day 0 and year 200 and 201 seem seem errors in set
            .loc[lambda x: (x.day != 0) & (x.year != 200) & (x.year != 201)]
            .drop(['state'], axis=1)
            .assign(region=lambda x: x.region.astype('category'),
                    country=lambda x: x.country.astype('category'),
                    city=lambda x: x.city.astype('category'))
            .assign(avgtemperature=(
                    lambda x: x.avgtemperature.fillna(method='ffill')))
            # convert Fahrenheit to Celcius
            .assign(temperature=lambda x: (x.avgtemperature-32)*5/9)
            .drop(['avgtemperature'], axis=1)
            .assign(date=lambda x: pd.to_datetime(x[['year', 'month', 'day']],
                    errors='coerce'))
            .drop(['year', 'month', 'day'], axis=1)
            # NOTE: you could also take the mean!
            .drop_duplicates(subset=index_names)
            .dropna()
            .set_index(index_names)
            .sort_index(level=index_names)
          )
    return df
