import pandas as pd
import datetime
import os

def save_data(data,
            data_dir,
            data_filename,
            assumptions,
            parameters,
            method,
            hyperparameters,
            derived_parameters,
            code_link = '',
            log_dir = './',
            log_filename='log.md',
            data_length = 2):
    '''
    Save the data in a file and write the metadata to a log file
    data: dictionary of arrays with the keys as the names of the columns of the data
    data_dir: directory to save the data file
    data_filename: name of the data file
    assumptions: list of strings describing the assumptions of the simulation
    parameters: dictionary of parameters of the simulation
    method: string describing the method used in the simulation
    hyperparameters: dictionary of hyperparameters of the simulation
    derived_parameters: parameters calculated from input parameters
    code_link: link to the code that generated the data
    log_dir: directory to save the log file
    log_filename: name of the log file
    data_length: length of the data to save
    '''
    # if the data directory does not exist, create it
    if not os.path.exists(data_dir):
        print(f"Creating directory {data_dir}")
        os.makedirs(data_dir)
    # make a data file name that is data_dir/data_filename
    data_file = data_dir + data_filename
    # concatenate the parameters to the data file name
    for key in parameters.keys():
        data_file += f"_{key}={parameters[key]}"
    # add .csv to the data file name
    data_file += '.csv'

    # The data is a dictionary of arrays where the keys are the names of the columns of the data
    # # first downsample the data so it has data_length elements
    for key in data.keys():
        step = max(1, len(data[key]) // data_length)
        data[key] = data[key][::step]
    df = pd.DataFrame(data)
    metadata = f"#Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    # the assumptions are in a list of strings
    metadata += "#Assumptions:\n"
    for assumption in assumptions:
        metadata += f"## {assumption}\n"
    # the parameters are in a dictionary
    metadata += "#Parameters:\n"
    for key in parameters.keys():
        metadata += f"## {key}: {parameters[key]}\n"
    metadata += "#Method:\n"
    metadata += f"## {method}\n"
    metadata += "#Hyperparameters:\n"
    for key in hyperparameters.keys():
        metadata += f"## {key}: {hyperparameters[key]}\n"
    metadata += "## Derived parameters:\n"
    for key in derived_parameters.keys():
        metadata += f"- {key}: {derived_parameters[key]}\n"        
    # write the metedata to the data file
    with open(data_file, 'w') as f:
        f.write(metadata)
        # print the data keys as the header of the data preceded by #
        f.write(f"#format:{' '.join(data.keys())}\n")
        format = ' '.join(['%.10e'] * len(data.keys()))
        df.to_csv(f, sep=' ', index=False, float_format='%.10e', header=False, mode='a')
        # print a confirmation message
        print(f"Data saved in {data_file}")
    metadata = f"## Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    # the assumptions are in a list of strings
    metadata += "## Assumptions:\n"
    for assumption in assumptions:
        metadata += f"- {assumption}\n"
    # the parameters are in a dictionary
    metadata += "## Parameters:\n"
    for key in parameters.keys():
        metadata += f"- {key}: {parameters[key]}\n"
    metadata += "## Method:\n"
    metadata += f"- {method}\n"
    metadata += "## Hyperparameters:\n"
    for key in hyperparameters.keys():
        metadata += f"- {key}: {hyperparameters[key]}\n"
    metadata += "## Derived parameters:\n"
    for key in derived_parameters.keys():
        metadata += f"- {key}: {derived_parameters[key]}\n"        
    # print link to the data_file in the metadata in md format
    metadata += f"## Data file: [{data_file}]({data_file})\n"
    # write the metadata to the log file
    log_file = log_dir + log_filename
    with open(log_file, 'a') as f:
        f.write(metadata)
        # print the code that generated the data
        f.write(f"## Code: [{code_link}]({code_link})\n")
        # print a horizontal line in md format
        f.write('---\n')
    # print a confirmation message
    print(f"Metadata saved in {log_file}")

def main():
    example_data = {
        'N': [1, 2, 3, 4, 5],
        'fid': [0.11231234, 0.2, 0.3, 0.4, 0.5],
        'ent': [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    save_data(
        data=example_data,
        data_dir='Results/test_saving_data/',
        data_filename='out_fid_ent_sweep',
        assumptions=['3D Green function perpendicular polarization'],
        parameters={'d': 1, 'kappa': 0.1, 'om': 1},
        method='example metadata writeout',
        hyperparameters={'dtype': 'float64', 'solver': 'mesolve', 'normalize_output': False},
        derived_parameters={},
        code_link='',
        log_dir='./',
        log_filename='log.md',
        data_length=2,
    )

if __name__ == '__main__':
    main()
