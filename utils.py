import numpy as np

from collections import OrderedDict

SEED = 1
applications = ['dns','ftp','http','ssdp','telnet']
powers_of_two = np.array([2**i for i in range(len(applications))])

# associate each feature with its min/max value
feature_list = OrderedDict([
    ('timestamp', [0,1<<32]),
    ('packet_length',[0,(1<<16)-1]),
    ('ttl',[0,(1<<8)-1]),
    ('TCP_length',[0,(1<<16)-1]),
    ('TCP_ack',[0,1<<32]),
    ('TCP_window_size',[0,(1<<16)-1]),
    ('src_port',[0,(1<<16)-1]),
    ('dst_port',[0,(1<<16)-1]),
    ('TCP_flags',[0,(1<<16)-1]),
    ('IP_flags',[0,(1<<16)-1]),
    ('application',[0,1<<len(applications)]),
    ]
)
# analize the dataset to find the min and max values of each feature
def find_min_max(X,time_window=10): 
    sample_len = X[0].shape[1] # number of features
    max_array = np.zeros((1,sample_len)) # initialize max_array with zeros
    min_array = np.full((1, sample_len),np.inf) # initialize min_array with inf

    for feature in X: # per ogni sample del dataset (il nome della variabile è fuorviante)
        temp_feature = np.vstack([max_array,feature]) # stack the max_array and the current feature
        max_array = np.amax(temp_feature,axis=0) # for each column, take the max value and update the max_array
        temp_feature = np.vstack([min_array, feature]) # we do the same for the min_array
        min_array = np.amin(temp_feature, axis=0)

    # flows cannot last for more than MAX_FLOW_DURATION seconds, so they are normalized accordingly
    #max_array[0] = time_window
    #min_array[0] = 0

    return min_array,max_array

# find the min and max values of each feature in the dataset BASING ON THE DICTIONARY, not on the dataset
def static_min_max(time_window):
    #feature_list['timestamp'][1] = time_window # we update the max value of the timestamp feature

    min_array = np.zeros(len(feature_list)) # initialize min_array with zeros
    max_array = np.zeros(len(feature_list)) # initialize max_array with zeros

    i=0
    for feature, value in feature_list.items(): # for each feature in the dictionary
        min_array[i] = value[0]
        max_array[i] = value[1]
        i+=1

    return min_array,max_array

# normalize the dataset
# rawpoints: dataset to normalize
# mins: min values of each feature
# maxs: max values of each feature
# high: max value of the normalized dataset
# low: min value of the normalized dataset
def scale_linear_bycolumn(rawpoints, mins,maxs,high=1.0, low=0.0):

    
    rng = maxs - mins
    ret = low+ (((high - low) * (rawpoints)) / rng)
    
    # Controlla se ci sono valori di ret fuori dai limiti
    if np.any(ret > high) or np.any(ret < low):
        print("Warning: Valori fuori dai limiti!")
        print(f"maxs: {maxs}")
        print(f"mins: {mins}")
        print(f"rawpoints: {rawpoints}")
        print(f"ret: {ret}")
    
    return ret

# inverse the normalization
# min=-1, max=1
def inverse_scale_linear_bycolumn_1(normalized_points, mins, maxs):
    normalized_points = np.array(normalized_points).astype(float)  # Converti in array di float
    mins = np.array(mins).astype(float)
    maxs = np.array(maxs).astype(float)
    
    rng = maxs - mins
    raw_points = (normalized_points + 1) * rng / 2 + mins
    
    # Gestisci i valori NaN
    raw_points = np.nan_to_num(raw_points, nan=0.0)

    # Gestisci i valori fuori intervallo
    raw_points = np.clip(raw_points, mins, maxs)

    return np.round(raw_points).astype(int)
    
# normalize the dataset and pad the samples to have the same length
# X: dataset to normalize
# mins: min values of each feature
# maxs: max values of each feature
# max_flow_len: length of the samples after padding
# padding: if True, pad the samples

def normalize_and_padding(X,mins,maxs,max_flow_len,padding=True):
    norm_X = []
    for sample in X:
        # 1) cut the sample if it is bigger than expected
        if sample.shape[0] > max_flow_len: 
            sample = sample[:max_flow_len,...]
        packet_nr = sample.shape[0] # number of packets in one sample
        # 2) normalize the sample
        norm_sample = scale_linear_bycolumn(sample, mins, maxs, high=1.0, low=-1.0)# normalize the sample
       
        # 3) remove NaN from the array
        np.nan_to_num(norm_sample, copy=False)  # NaN is replaced by zero and infinity by large finite numbers, in place
        # 4) pad the sample if it is smaller than expected
        if padding == True:
            # padding the sample with zeros on the rows
            norm_sample = np.pad(norm_sample, ((0, max_flow_len - packet_nr), (0, 0)), 'constant',constant_values=(0, 0))  # padding
        if np.any(norm_sample > 1) or np.any(norm_sample < -1):
            print("Warning: Valori fuori dai limiti!")
            print(f"maxs: {maxs}")
            print(f"mins: {mins}")
            print(f"rawpoints: {sample}")
            print(f"norm_sample: {norm_sample}")
        norm_X.append(norm_sample)
    return norm_X

# count the number of packets in each sample of the dataset
# the function makes sense if features have only positive values 
def count_packets_in_dataset(X_list):
    packet_counters = []
    for X in X_list: # for each sample in the dataset
        TOT = X.sum(axis=1) # sum the values of the features in each row
        packet_counters.append(np.count_nonzero(TOT)) # count the number of non-zero values (i.e. the number of packets) and append it to the list

    return packet_counters