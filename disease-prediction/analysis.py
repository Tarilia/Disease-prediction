def analyze_the_data(df):
    print(f'Looking through the first ten entries \n{df.head(10)}')
    print('\nStudying the data structure')
    df.info()
    print(f'\nCheck for any null values \n{df.isnull().sum()}')
    print(f'\ngetting a statistical description of the data \n{df.describe()}')
