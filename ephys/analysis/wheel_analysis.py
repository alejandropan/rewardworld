from os import F_TEST


#g_t  go cue trigger time
#f_t feedback trigger time
#pos position file
#pos_t position timestamps file

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def find_nearest(array, values):
    nearests= []
    array = np.asarray(array)
    try:
        for i in values:
            nearests.append(np.abs(array - i).argmin())
    except:
        nearests=np.abs(array - values).argmin()
    return nearests

maximum_displacement = []
for t in np.arange(len(g_t)):
    timestamps = np.where((pos_t>g_t[t])&(pos_t<f_t[t]))
    pos_0 = pos[timestamps][0]
    max_d = np.max(abs(pos[timestamps]-pos_0))
    maximum_displacement.append(max_d)

start= 154
stop = start+4
timestamps = np.where((pos_t>g_t[start])&(pos_t<f_t[stop+1]))

plt.plot(pos_t[np.where((pos_t>g_t[start])&(pos_t<f_t[stop+1]))], pos[timestamps], color='k')
plt.scatter(g_t[start:stop], pos[find_nearest(pos_t,g_t[start:stop])], color='g')
plt.scatter(f_t[start:stop], pos[find_nearest(pos_t,f_t[start:stop])], color='r')
plt.xlabel('Seconds')
plt.ylabel('Position(rad)')


deltas = pos[find_nearest(pos_t,f_t[start:stop])]-pos[find_nearest(pos_t,g_t[start:stop])]

start_dots = []
finish_dots = []

for i,trial in enumerate(np.arange(start,stop)):
    p0 = pos[find_nearest(pos_t,g_t[trial])]
    dpos = pos-p0
    if i==0:
        start_dots.append(dpos[find_nearest(pos_t,g_t[trial])])
        finish_dots.append(dpos[find_nearest(pos_t,f_t[trial])])
        data = dpos[find_nearest(pos_t,g_t[trial]):find_nearest(pos_t,f_t[trial])]
        data_t = pos_t[find_nearest(pos_t,g_t[trial]):find_nearest(pos_t,f_t[trial])]
    else:
        start_dots.append(dpos[find_nearest(pos_t,g_t[trial])])
        finish_dots.append(dpos[find_nearest(pos_t,f_t[trial])])
        datat = dpos[find_nearest(pos_t,g_t[trial]):find_nearest(pos_t,f_t[trial])]
        datat_t = pos_t[find_nearest(pos_t,g_t[trial]):find_nearest(pos_t,f_t[trial])]
        data = np.concatenate([data,datat])
        data_t= np.concatenate([data_t,datat_t])

plt.plot(data_t,data)
plt.scatter(g_t[start:stop], start_dots, color='g')
plt.scatter(f_t[start:stop], finish_dots, color='r')
plt.xlabel('Seconds')
plt.ylabel('Position(rad)')