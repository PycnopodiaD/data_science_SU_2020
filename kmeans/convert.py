def convert_data_to_csv(data_path, csv_path):
    """
    Function that locates the data and converts the .data format file, into
    a .csv format file.

    To accomplish this make two varaibles called data_path and csv_path with the
    path to the file to be converted (.data) and the file that should be
    produced (.csv) using the os.path.realpath implimentation of the os python
    library.

    Parameters
    ----------
    data_path : str
        Location of the .data format file
    csv_path : str
        Location of the .csv format file
    Returns:
        None
    ----------
    """
    df = pd.DataFrame(columns=['col'+str(i+1) for i in range(12)])
    with open(data_path, 'r') as fr:
        while True:
            line = fr.readline()
            if len(line) == 0:
                break
            row = {'col'+str(i+1): int(val) for i, val in enumerate(line[:-1].split())}
            df = df.append(row, ignore_index=True)
    df.to_csv(csv_path, index=False)
