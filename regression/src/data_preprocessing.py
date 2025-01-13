import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

vehicle_class_mapping = {
    'COMPACT': 12.305747126436781,
    'FULL-SIZE': 14.452173913043477,
    'MID-SIZE': 13.6828125,
    'MINICOMPACT': 14.677777777777777,
    'MINIVAN': 15.885714285714286,
    'PICKUP TRUCK - SMALL': 14.11,
    'PICKUP TRUCK - STANDARD': 17.370666666666665,
    'STATION WAGON - MID-SIZE': 13.24,
    'STATION WAGON - SMALL': 11.963636363636363,
    'SUBCOMPACT': 12.088157894736844,
    'SUV': 15.984615384615385,
    'TWO-SEATER': 16.928571428571427,
    'VAN - CARGO': 18.011764705882353,
    'VAN - PASSENGER': 19.323076923076922
}

make_mapping = {
    'ACURA': 12.557142857142859,
    'AUDI': 14.36111111111111,
    'BMW': 14.795454545454545,
    'BUICK': 13.9,
    'CADILLAC': 15.233333333333334,
    'CHEVROLET': 15.418750000000001,
    'CHRYSLER': 13.893749999999999,
    'DAEWOO': 11.81111111111111,
    'DODGE': 18.762162162162163,
    'FERRARI': 25.200000000000003,
    'FORD': 15.908510638297873,
    'GMC': 17.651724137931033,
    'HONDA': 11.214285714285714,
    'HYUNDAI': 12.0,
    'INFINITI': 13.525,
    'ISUZU': 15.642857142857142,
    'JAGUAR': 15.814285714285715,
    'JEEP': 16.45714285714286,
    'KIA': 12.275,
    'LAND ROVER': 20.225,
    'LEXUS': 15.183333333333332,
    'LINCOLN': 16.45,
    'MAZDA': 13.575,
    'MERCEDES-BENZ': 14.04,
    'NISSAN': 15.225,
    'OLDSMOBILE': 13.38,
    'PLYMOUTH': 15.4,
    'PONTIAC': 12.515789473684212,
    'PORSCHE': 15.557142857142859,
    'SAAB': 13.685714285714285,
    'SATURN': 11.014285714285714,
    'SUBARU': 12.166666666666666,
    'SUZUKI': 10.830769230769231,
    'TOYOTA': 12.975675675675674,
    'VOLKSWAGEN': 11.810344827586206,
    'VOLVO': 13.664705882352942
}

transmission_mapping = {
    'A3': 12.777777777777779,
    'A4': 15.266015625,
    'A5': 14.944642857142856,
    'AS4': 14.4,
    'AS5': 15.8,
    'M5': 13.024050632911393,
    'M6': 17.01875
}



def preprocess(dataset_path):

    dataset = pd.read_csv(dataset_path)

    # drop unnecessary columns
    dataset.drop(columns=['Year', 'MODEL', 'FUEL'], inplace=True)


    # mean encoding
    dataset['VEHICLE CLASS'] = dataset['VEHICLE CLASS'].map(vehicle_class_mapping)
    dataset['MAKE'] = dataset['MAKE'].map(make_mapping)
    dataset['TRANSMISSION'] = dataset['TRANSMISSION'].map(transmission_mapping)
    df_numeric = dataset.select_dtypes(include=[np.number])

    # return the processed dataset
    return df_numeric