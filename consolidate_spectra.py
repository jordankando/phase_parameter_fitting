import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def consolidate_spectra(base_dirs, ID = ""):

    if ID is not "":
        ID = "_"+ ID

    if not isinstance(base_dirs, str):

        files = []
        which_date = []

        for idx, base_dir in enumerate(base_dirs):
            files_in_dir = [base_dir + i for i in os.listdir(base_dir)]
            which_date += [int(base_dir[3:-1])] * len(files_in_dir)
            files += files_in_dir
    else:
        files = []
        which_date = []

        base_dir = base_dirs
        files_in_dir = [base_dir + i for i in os.listdir(base_dir)]
        date_index = base_dir.find("230")
        print(base_dir[date_index:date_index+6])
        which_date += [int(base_dir[date_index:date_index+6])] * len(files_in_dir)
        files += files_in_dir


    mask = [files[i][-3:] == "txt" for i in range(len(files))]

    files = np.array(files)

    spectra_files = files[mask]
    which_date = np.array(which_date)[mask]
    print(spectra_files)

    test_file = spectra_files[0]

    df = pd.read_csv(test_file, delimiter = '\t')
    wl = df["Wavelength"].to_list()
    wl_str= [str(i) for i in wl]

    print(wl)

    col_names = ['Phase Angle'] + wl_str + ['Date Taken']
    print(col_names)
    print(df.iloc[:,1].to_list())

    df_spectralon = pd.read_excel("/Users/jordanando/Documents/Research/Li Group Research/Labwork/SpectralonCorrectionFactors.xlsx")

    spectralon_angle = df_spectralon['Emission angle'].to_numpy()
    spectralon_corr= df_spectralon['Correction factor'].to_numpy()

    def compute_spectralon_correction(e):
        return np.interp(e, spectralon_angle[::-1], spectralon_corr)


    df_rows = []



    for idx, file in enumerate(spectra_files):
        print(file)
        df_file = pd.read_csv( file, delimiter = '\t')

        e_index = file.find("i44_e") + 4

        row_entries = []
        if file[e_index + 1] == "n":
            e = -int(file[e_index + 2:e_index+4])
            row_entries.append((e+45))
            
        else:
            e = int(file[e_index + 1:e_index+3])
            row_entries.append(e+45)
        print("e: ", e)
        print("g: ", row_entries[0])

        spectrum = df_file.iloc[:,1].to_numpy()

        spectrum *= compute_spectralon_correction(e)
        print(compute_spectralon_correction(e))

        row_entries += spectrum.tolist()
        print(which_date[idx])
        print(idx)
        row_entries.append(which_date[idx])


        df_rows.append(row_entries)

        df_spectra = pd.DataFrame(df_rows, columns=col_names)

    df_spectra.sort_values(by = ['Phase Angle'], inplace = True)
    #display(df_spectra)

    df_spectra.reset_index(inplace=True, drop = True)


    df_spectra.to_csv(base_dir[date_index:date_index+6] + "_" + ID + "_Spectra_Highlands_Simulant_CORRECTED.csv")

    gs = np.unique(df_spectra["Phase Angle"])
    mean_700 = []
    std_700 = []

    print(gs)

    for idx, g in enumerate(gs):
        meas = df_spectra[df_spectra["Phase Angle"] == g]['700']
        #display(meas)

        mean_700.append(np.mean(meas))
        std_700.append(np.std(meas))

    print(mean_700)

    rows = [gs.tolist(), mean_700, std_700]

    rows = np.array(rows).T.tolist()

    print(rows)
    cols = ["Phase Angle", 700, "std"]
        
    df_mean = pd.DataFrame(rows, columns=cols)
    df_mean.sort_values(by = ['Phase Angle'], inplace = True)
    #display(df_spectra)

    df_mean.reset_index(inplace=True, drop = True)
    

    df_mean.to_csv(base_dir[date_index:date_index+6] + ID + "_Spectra_Highlands_Simulant_MEAN.csv")

    return base_dir[date_index:date_index+6]  + ID + "_Spectra_Highlands_Simulant_MEAN.csv"


if __name__ == "__main__":

    if len(sys.argv) > 2:
        outfile = consolidate_spectra(sys.argv[1], ID = sys.argv[2])
    else:
        outfile = consolidate_spectra(sys.argv[1])

    
    

    print(outfile)