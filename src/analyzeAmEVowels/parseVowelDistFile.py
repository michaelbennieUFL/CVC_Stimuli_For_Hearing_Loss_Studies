import pandas as pd


def parse_vowel_dist_data(file_path='../../input_data/vowelDistData.txt'):
    import pandas as pd
    # Initialize a dictionary to hold data
    data_dict = {}
    measures = ['Duration', 'F0', 'F1', 'F2', 'F3', 'F4']

    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            if len(tokens) < 8:
                continue
            vowel, group, mean, sd, *_, measure = tokens
            if measure not in measures:
                continue
            key = (vowel, group)
            if key not in data_dict:
                data_dict[key] = {}
            data_dict[key][f"{measure.lower()}_mean"] = float(mean)
            data_dict[key][f"{measure.lower()}_sd"] = float(sd)

    # Build DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index, names=['vowel', 'group'])
    df = df.reset_index()
    return df

if __name__=='__main__':
    df = parse_vowel_dist_data('../../input_data/vowelDistData.txt')
    print(df.head())
