import pickle



def get_datas(file, data_field): 
    '''
    Function to load in an hdf5 file and access specific data from that file. This will loop through all of the events, and return a dictionary of arrays 
    correpsonding to the data captured for that particular event.
    '''
    datas = {}
    for event_id in file.keys():
        data = file[event_id][data_field][()]
        datas[event_id] = data
        
    return datas

def get_data(file, data_field, event_id): #only pulls a single event from an hdf5 file
    '''
    This function will look in an hdf5 file and load in a single time series corresponding to a single event and return the array.
    '''
    data = file[event_id][data_field][()]
    return data

def save_results(results, filename):
    '''
    Function to pickle results and save them
    '''
    with open(filename, 'wb') as file:
        pickle.dump(results, file)
    print(f"Results successfully saved to {filename}")

def load_results(filename):
    '''
    Function to load pickled results
    '''
    try:
        with open(filename, 'rb') as file:
            results = pickle.load(file)
            print(f"Results successfully loaded from {filename}")
        return results
    except FileNotFoundError:
        return []
    
