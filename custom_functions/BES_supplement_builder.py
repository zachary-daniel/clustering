'''
This is a file for pulling data from the h5 files in CETOP/BES_supplement . All of the data that is on the database descriptions doc will be pulled from the files, and interpolated onto the 
BES diagonstics time base. This will then all be put into one h5 file. Not sure what I'll call it yet but I guess we'll see. Since all of these h5 files are labeled by shot number, we will
also have to chop them up and time them with the ELM events from the corresponding shots. 

The file suffixes used in the google doc are
_basic, _actu, _mag, _mag_hi, _co2_s, and _ece. co2_s is labeled as c02_den in the google doc. co2_s contains the same data as listed on the doc. I haven't seen any files that reference mag_hi,
and for some reason some of the magnetics data is stored in magentics.pkl, and some in _mag.h5. We'll have to deal with that
'''
import numpy as np
import h5py
import time
import data_processing_functions as dpf
import multiprocessing
import pickle

def get_shots(shot_nums_path, num_events):
    shot_nums = np.zeros(num_events) #which shot corresponds to which event id
    
    with open(shot_nums_path, 'r') as shots: #read text file
        for index,shot in enumerate(shots): #cycle through all entries and add to a list of all shots for each event, and a list of unique shots
            shot_nums[index] = shot[:-1] #skip the last character for each entry. Just a new line '/n'
    unique_shots = np.unique(shot_nums)
    return shot_nums, unique_shots

#function for passing the data dictionary around and writing to it with multiprocessing
def init_worker( bes_keys, shot_nums):
    global _bes_keys_
    _bes_keys_ = bes_keys
    global _shot_nums_
    _shot_nums_ = shot_nums

def data_from_shot(shot_num):
    supplemental_data_dict = {}
    elms_in_shot = np.where(_shot_nums_ == int(shot_num))[0] #find the position of the event ids that correspond to the current shot
    ids = [bes_keys[index] for index in elms_in_shot] #select the event ids that are from the shot
    #select the bes signals that correspond to the events that are from the current shot. This is a list of arrays
    bes_time_list = [bes_times[id] for id in ids]#time of all the time arrays for the corresponding BES signals
    xdata_dict = {} #empty dicts for x, y, z data. data from the hdf5 files will be stored in here
    zdata_dict = {}
    ydata_dict = {}
    for suffix in file_suffixes: #cycle over all of the data files
        data_supplement_path = BES_supplement_path + '/' + str(int(shot_num)) + suffix + '.h5' #construct path to particular data file
        try: #try opening the h5 file if it exists. Not al shots have high frequency magnetics at the moment
            with h5py.File(data_supplement_path, 'r') as supplement_file: #
                supplement_file.keys()
        except:
            continue
        with h5py.File(data_supplement_path, 'r') as supplement_file: #open the h5 file
            for key in supplement_file.keys(): #cylce over all the keys in the h5 file
                if key in signal_names and key != 'ece': #check if the current key is in the list of desired signals
                    zdata = np.asarray(supplement_file[key]['zdata']) #assign zdata. If this fails, it's probably from the ece files. They have a different structure for some reason
                    print(len(zdata))
                    xdata = np.asarray(supplement_file[key]['xdata']) #get the x component of data
                    
                    try: 
                        len(xdata) #this means no signal was recorded so we'll skip this key
                    except:
                        continue
                    #if the data was actually recorded, we'll add it to the dictionary
                    xdata_dict[key] = xdata
                    zdata_dict[key] = zdata
                    ydata = -1 #place holder value for ydata if this field is empty
                    if supplement_file[key]['ydata'] != -1: #if there is not ydata, this field is -1 in the h5 file
                        ydata = np.asarray(supplement_file[key]['ydata']) #assign if there is actual ydata
                        ydata_dict['key'] = ydata
                        
                elif key == 'ece': #the ece files have a different format we need to account for.
                    key_base = 'tecef' 
                    for i in range(40):
                        num = i+1
                        if num < 10: #tecef signals are headered as tecef01,tecef02, .... tecef40. need to account for the leading zero for channel < 10
                            num_str = '0' + str(num)
                        else:
                            num_str = str(num)
                        data_str = key_base + num_str #completed string for tecef data
                        zdata = np.asarray(supplement_file[key][data_str])
                        zdata_dict[data_str] = zdata
                        xdata = np.asarray(supplement_file[key]['xdata']) #get time vector
                        xdata_dict[data_str] = xdata #assign to the same key as zdata
                else: #key is not in the list
                    pass
    
    #cycle over all of the elms in the shot
        for index, id in enumerate(ids):
            bes_time_id = bes_time_list[index] #time for the given id
            supplemental_data_dict[id] = {} #assign an empty dicitonary to the current event id
            for key in zdata_dict.keys(): #cycle over all of the data
                #it seems that some of this data also just fully doesn't exist. We'll put in a check for that
                zdata_shot = zdata_dict[key] #select the z and x data. xdata is always time in ms
                xdata_shot = xdata_dict[key]
                ydata_exists = True
                try:
                    ydata_shot = ydata_dict[key]
                    ydata_interp = np.interp(bes_time_id, xdata_shot, ydata_shot)
                except:
                    ydata_exists = False
                
                zdata_interp = np.interp(bes_time_id, xdata_shot, zdata_shot) #takes the time base you want to sample onto, the original time base, and the original signal as inputs
                # if ydata_shot != -1: #if there is ydata that is not just a placeholder value, interpolate it onto the bes time base
                    # ydata_interp = np.interp(bes_time_id, xdata_shot, ydata_shot)
                if ydata_exists:
                    supplemental_data_dict[id][key] = {'zdata': zdata_interp, 'ydata': ydata_interp}
                    print('There is ydata!')
                else:
                    supplemental_data_dict[id][key] = zdata_interp
    return supplemental_data_dict

def get_data_dict(shot_nums, unique_shots, bes_keys):
    processes = 30 #number of threads. can be changed by user
    with multiprocessing.Pool(processes = processes, initializer=init_worker, initargs=(bes_keys,shot_nums,)) as pool:
        data_dict_list = pool.map(data_from_shot, unique_shots)

        
     #cycle through all of the shot numbers 
    return data_dict_list
#write save dicitonary as npy file
# np.save('custom_functions/supplemental_dict.npy', supplemental_dict, allow_pickle=True)
# hf = h5py.File('supplemental_data_test.h5', 'w')
# hf.attrs.update(supplemental_dict.astype(np.float64))
# hf.close()


#I should probably turn this into a script with methods and all that shit...
if __name__ == '__main__':
    start = time.monotonic()
    BES_supplement_path = '../../CETOP/BES_supplement' #path to the h5 files with the supplemental data
    BES_data_path = '../../CETOP/BES_DIIID/labeled_elm_events.hdf5' #path to file containing BES signals. 
    with h5py.File(BES_data_path, 'r') as file:
        bes_signals = dpf.get_datas(file, 'signals') #dictionary with each key as an elm event, each value as a time series of BES data
        bes_labels = dpf.get_datas(file, 'labels') #labels for when an elm is occuring
        bes_times = dpf.get_datas(file, 'time')  #time base for each elm event

    file_suffixes = ['_basic', '_actu', '_co2_s', '_ece', '_mag_hi']
    signal_names = ['bt','echpwrc','fs00','fs01','fs02','fs03','fs04','fs05','ip','pinjf_15l','pinjf_15r','pinjf_21l',
                    'pinjf_21r','pinjf_30l','pinjf_30r','pinjf_33l','pinjf_33r','b1', 'b2', 'b3','b4','b5', 'b6', 'b7', 'b8', 'r0','v1','v2','v3', 'ece'] #list of desired signals
    shot_nums_path = '../shot_nums.txt' #text file containing the shot number for each ELM event. Ordered by increasing elm event id. 
    num_events = len(bes_signals.keys()) #number of events
    bes_keys = list(bes_signals.keys()) #keys for the bes signals are the event id labels i.e. '00000', '00003' etc.

    shot_nums, unique_shots = get_shots(shot_nums_path, num_events) #get the shot numbers that will be used for reading the h5 files and so on
    data_dict_list = get_data_dict(shot_nums, unique_shots, bes_keys) #generate dictionaries for each shot with the supplemental data. Each one should
    supplemental_data_dict = {}
    for dict in data_dict_list:
        for key in dict.keys():
            supplemental_data_dict[key] = dict[key]
    print(f'there are {len(supplemental_data_dict.keys())} keys')

    # h5_save_path = '../bes_supplement_data/supplemented_data_test.hdf5'
    # h = h5py.File(h5_save_path, 'w')
    # for key in supplemental_data_dict.keys():
    #     group = h.create_group(key)
    #     for data_set_name in supplemental_data_dict[key]:
    #         data = group.create_dataset(data_set_name, data = supplemental_data_dict[key][data_set_name])
    # h.close()
    # print(f'File successfully written to {h5_save_path}')