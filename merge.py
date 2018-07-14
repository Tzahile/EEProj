import math
#import random
#import matplotlib.pyplot as plt
import numpy as np
#import operator


def diff_py(matrix):
    matrix_diff = np.transpose(np.diff(np.transpose((matrix))))
    return matrix_diff


def diff_and_find_py(matrix):
    matrix_diff = np.transpose(np.diff(np.transpose((matrix))))
    matrix_diff_arr = np.asarray(matrix_diff).reshape(-1)
    find_matrix = np.nonzero(matrix_diff_arr)
    result = np.transpose(find_matrix)
    return result


def find_py(matrix):
    matrix_arr = np.asarray(matrix).reshape(-1)
    find_matrix = np.nonzero(matrix_arr)
    result = np.transpose(find_matrix)
    return result


def tran_detect_debug():
    f_sys = 50
    sample_per_cycle = 120
    noise_th = 2
    # max_transient_t = 0.01; #t for time
    steady_t = 0.02
    # max_transient_s = max_transient_t*(f_sys*sample_per_cycle);

    # =======debug only=============
    t_debug = np.linspace(0, 20./f_sys, (20. * sample_per_cycle + 1))
    #step = (1.0/(f_sys*sample_per_cycle))
    #t_debug = np.arange(0, (20.0/f_sys + step), step)
    v_t = 220*np.cos(t_debug*2*math.pi*f_sys)  # +-10+20*rand(1,len(t_debug))
    # for i=1:4
    #     v_t(i*100+200:i*100+220)=5;
    # v_t(400:600)=40*rand(1,200+1)+v_t(400:600);
    # v_t(1000:1200)=100*rand(1,200+1)+v_t(1000:1200);
    # v_t(200:201) = 250;
    # v_t(150:200) = 10*cos(t_debug(150:200)*2*10*pi*f_sys);
    newTDebug = t_debug[0: (len(v_t)-349)]*4*np.pi*f_sys+math.pi
    cosVar = np.cos(newTDebug)
    v_t[349:len(v_t)] = v_t[349:len(v_t)] + 100 * cosVar

    v_t[349:380] = 1
    v_t[799:810] = 350
    v_t[999:1050] = -20+40*np.random.random((1, 51))
    v_t[1399:] = v_t[1399:] + 220 * \
        np.cos(t_debug[0:(len(v_t)-1399)]*2*math.pi*2*f_sys+math.pi)
    v_t[1399:1410] = 1
    v_t[1799:1850] = -20+40*np.random.random((1, 51))
    v_t[2099:] = -50+100*np.random.random((1, len(v_t)-2099))
    # v_t(3400:3601)=100*rand(1,3601-3400+1);
    # =============================

    res_trans = cpc_trans_detect_main(
        v_t, f_sys, sample_per_cycle, noise_th, steady_t)
    plt.subplot(2, 1, 2)
    plt.plot(t_debug*1000, v_t, linewidth=2.0)
    plt.xlabel('Time [mSec]')
    plt.ylabel('Input Signal - V[n]')
    plt.title('Simulation #11 - Several Loads and Transients')
    plt.subplot(2, 1, 2)
    plt.plot(t_debug*1000, res_trans, linewidth=2.0)
    plt.xlabel('Time [mSec]')
    plt.ylabel('Output Signal - R[n]')
    plt.ylim([-0.1, 2.1])


# **********************************************
# **********************************************


def cpc_trans_detect_main(v_t, f_sys, sample_per_cycle, noise_th, steady_t):
    # first iteration - find transients
    res_trans = tran_detect(v_t, f_sys, sample_per_cycle, noise_th, steady_t)
    # Assuming that if end of sample is transient, there is new load(s), run
    # load_detect function.
    while (res_trans[0, len(v_t)-1] == 1):
        [res_trans, load_stable] = load_detect(
            v_t, sample_per_cycle, noise_th, res_trans)
        if(load_stable > (len(v_t)-2*sample_per_cycle)):
            break

       # After finding new load run new iteration and find additinal
       # transients based on new load.
    res_trans[load_stable-1:] = tran_detect(
        v_t[load_stable-1:], f_sys, sample_per_cycle, noise_th, steady_t)

    # Using transient results and create log table
    log_matrix = tran_log(res_trans, v_t, f_sys, sample_per_cycle)
    # print table to file
    print_log(log_matrix)

    return res_trans

# *************************************************************************************************************************************************
# *************************************************************************************************************************************************


def load_detect(v_t, sample_per_cycle, noise_th, res_trans):

    # this function will handle detection of new load
    # this function assume first cycle contain transient due to load connection
    load_stable = len(v_t)
    # this while will find the first non - trans sample before the long transient
    while res_trans[0, load_stable-1] > 0:
        load_stable = load_stable - 1

    load_stable = load_stable + 1
    new_trans = load_stable

    # This part find end of transient that occurred during new load connection.
    delta = 0
    cycles_not_equal = True
    while cycles_not_equal and load_stable < (len(v_t)-sample_per_cycle):
        if load_stable+2*sample_per_cycle+1 > len(v_t):
            return  # assuming two last cycles are part of transient.

        # compare two neighbor cycles in order to decide when load transient is over.
        delta_tran = delta
        firstVT = v_t[load_stable-1: load_stable + sample_per_cycle - 1]
        secVT = v_t[load_stable + sample_per_cycle -
                    1: load_stable + 2*sample_per_cycle - 1]
        delta = np.abs(np.subtract(firstVT, secVT))
        if (delta < noise_th):
            cycles_not_equal = False
        else:
            # new trans is beginning of cycle. two cycles are not the same therefor
            # load transient isn't over.
            # at loop end load_stable indicate first stable cycle.
            load_stable = load_stable + sample_per_cycle

    # find the sample when transient due to new load is over
    temp = find_py(abs(delta_tran) <= noise_th)
    # define when the load transient is over
    load_stable = load_stable-(sample_per_cycle-temp[0]+1)
    temp = None
    # define the samples indicates new load transients
    res_trans[new_trans:load_stable] = 2
    # define rest of results vector 0 for next iteration
    res_trans[load_stable:len(v_t)] = 0
    return [res_trans, load_stable]

# *************************************************************************************************************************************************
# *************************************************************************************************************************************************


def print_log(log_matrix):
    file = open("Transient_Log _Summary.txt", "w")
    file.write("========== Transient Log ==========\r\n")
    # ***file.write("==========  %7s  ==========\r\n" % date)
    file.write("=============================\r\n")
    file.write("\r\n")
    file.write("There are %d transients in the sample.\r\n")
    # ***file.write('There are %d transients in the sample.\r\n' % sum(operator.ne(log_matrix[:1], 0))
    # ***file.write('There are %d transients due to a disturbance. \r\n' % sum(log_matrix[:1] == 1))
    # ***file.write('There are %d transients due to change of load. \r\n' % sum(log_matrix[:1] == 2))
    file.write('\r\n')
    # np.transpose(np.nonzero(x))
    # tran_log_index=find(log_matrix[:, 0] ~=0)  # need to be fixed
    tran_log_index = find_py(log_matrix[:, 0])

    for i in range(0, len(tran_log_index)):
        file.write('Transient #%d\r\n' % i)
        file.write('========\r\n')
        if log_matrix[tran_log_index(i)-1:1] == 1:
            file.write('This transient occurred due to a disturbance.\r\n')
        else:
            file.write('This transient occurred due to change of load.\r\n')
        # for all time values we are using msec units.
        file.write(
            'This transient started after %.2f[msec] from the beginning of the sample.\r\n' % log_matrix[tran_log_index[i]-1:2]*1e3)
        file.write(
            'This transient ended after %.2f[msec] from the beginning of the sample.\r\n' % log_matrix[tran_log_index[i]-1:3]*1e3)
        if log_matrix[tran_log_index[i]:4] > 10e-6:
            file.write(
                'Transient duration time is %.2f[msec].\r\n' % log_matrix[tran_log_index[i]-1:4]*1e3)
        else:
            file.write(
                'Transient duration time is %.2f[usec].\r\n' % log_matrix[tran_log_index[i]-1:4]*1e6)
        if log_matrix[tran_log_index[i]-1:1] == 1:
            file.write(
                'This transient has started %.2f[msec] after the end of previous transient that occurred due to a disturbance.\r\n' % log_matrix[tran_log_index[i]-1, 10]*1e3)
        # ***file.write('The RMS value of this transient is %.2f[V].\r\n', log_matrix(tran_log_index(i), 5))# need to be fixed

        # ***file.write('The relative RMS value of this transient compared to steady state is %.2f%s.\r\n' % log_matrix[tran_log_index[i]:6])*100, '%')
        # ***file.write('The relative Vpeak of this transient compared to steady state is %.2f%s.\r\n' %log_matrix[tran_log_index[i]:7]*100, '%')
        if tran_log_index[i]-1 == len(log_matrix[:, 0]):
            file.write(
                'This transient lasts until the end of sample.\r\nPlease check if there is a problem in the power grid.\r\n')
        file.write('\r\n')

    file.close()

# *************************************************************************************************************************************************
# *************************************************************************************************************************************************


def tran_detect(v_t, f_sys, sample_per_cycle, noise_th, steady_t):
    # comparing samples start from second cycle (No.1)
    sample_counter = sample_per_cycle - 1

    # define reference cycle to compare.
    # assuming first cycle is not a transient

    ref_cycle = v_t[0: sample_per_cycle]

    # minimum time/samples between two different transients
    steady_s = steady_t*(f_sys*sample_per_cycle)

    # vector results eqauls 0 if not part of transient, 1 for part of transient.
    res_trans = np.zeros((1, len(v_t)))

    # this loop marks each sample that belong to any transient
    while sample_counter + 1 <= len(v_t):
        # we will use modulo on the reference cycle vector and compare samples
        ref_index = (sample_counter + 1) % sample_per_cycle
        # in case modulo equals 0 this is the last sample and compare to end of refernce cycle.
        # if ref_index == 0:
        #     ref_index = sample_per_cycle

        # calculate delta between samples
        delta = abs(v_t[sample_counter] - ref_cycle[ref_index-1])
        # if difference is smaller than noise_th+1e-6,then this sample isn't part of transient, continue. The value 1e-6 fixing matlab accuracy issues in case noise_th = 0
        if delta <= noise_th + 1e-6:
            sample_counter += 1
            continue
        # the sample is part of a transient, mark 1 in results vector.
        res_trans[0, sample_counter] = 1
        sample_counter += 1

    # This part of the code dealing with min time between transients.
    # The user define min time between transients (steady_t).
    # If the transients are too close then we define this a one long transient.

    sample_counter = sample_per_cycle+1
    while sample_counter < len(v_t) and res_trans[0, sample_counter] == 0:
        sample_counter = sample_counter + 1

    start = sample_counter

    # this loop check and fix steady state assumption between transients
    while sample_counter < np.size(v_t):

        while sample_counter < len(v_t) and res_trans[0, start] == 1:
            start = start+1
            sample_counter = sample_counter + 1

        stop = start
        while sample_counter < len(v_t) and res_trans[0, stop] == 0:
            stop = stop + 1
            sample_counter = sample_counter + 1

        if sample_counter > len(v_t):
            break
        if (stop-start) < steady_s:
            res_trans[0, start: stop-1] = 1
        start = sample_counter
    return res_trans

# *************************************************************************************************************************************************
# *************************************************************************************************************************************************


def tran_log(res_trans, v_t, f_sys, sample_per_cycle):

    # ______vector contains all start and end indexes of all transients______
    trans_index = diff_and_find_py(res_trans)
    #trans_index = find(np.diff(res_trans) != 0)
    trans_index[len(trans_index)+1] = len(v_t)
    # ______define matrix with data for log______
    log_matrix = np.zeros((len(trans_index), 10))
    # ______log_matrix columns defenition______
    #  1           2              3                   4                 5           6                     7              8               9         10
    # [type]  [start_time(sec)] [end_time(sec)] [duration_time(sec)] [RMS(V)] [tran RMS relative] [Vpeak relative] [start_index] [end_index]   [Time between transients}
    log_matrix[:, 8] = trans_index
    log_matrix[0, 7] = 1  # first index
    log_matrix[1:len(trans_index), 7] = log_matrix[1:len(
        trans_index)-1, 9]+1  # create start index column
    cycles = len(v_t)/sample_per_cycle
    t = np.linspace(0, 1, 1)/np.linspace((f_sys*sample_per_cycle),
                                         cycles)/f_sys-1/(f_sys*sample_per_cycle)
    # ______convert from index axis to time______
    log_matrix[:, 2:3] = t(log_matrix[:, 8:9])
    log_matrix[:, 4] = log_matrix[:, 3] - \
        log_matrix[:, 2]  # calculate duration time
    log_matrix[:, 0] = res_trans(log_matrix[:, 8])  # insert type from results

    # ______RMS calculation based on discrete RMS calculation RMS=sqrt(sum(V^2)/num of samples)______
    for i in range(1, len(trans_index)):
        log_matrix[i, 5] = math.sqrt(sum(v_t[log_matrix[i, 8]:math.pow(
            log_matrix[i, 9], 2)]/(log_matrix[i, 9]-log_matrix[i, 8]+1)))
    # Calculate reltive RMS of transients in compare to steady state and calculate Vpeak relative to steady state

    i = 1
    # loop handle all transient except transient in the end of the signal if exists
    while i < len(trans_index):
        if(log_matrix[i, 1] != 0):
            log_matrix[i, 6] = log_matrix[i, 5]/log_matrix[i+1, 5]
            log_matrix[i, 7] = max(abs(v_t[log_matrix[i, 8]:log_matrix[i, 9]])) / \
                max(abs(v_t[log_matrix[i+1, 8]:log_matrix[i+1, 9]]))
        i = i+1
    # handle a transient at the end of signal in case it exists
    if log_matrix[len(trans_index), 1] != 0:
        i = len(trans_index)
        log_matrix[i, 6] = log_matrix[i, 5]/log_matrix[i-1, 5]
        log_matrix[i, 7] = max(abs(v_t[log_matrix[i, 8]:log_matrix[i, 9]])) / \
            max(abs(v_t[log_matrix[i-1, 8]:log_matrix[i-1, 9]]))

    # calculate time between transients, only type 1 transient
    tran_log_index = np.where(log_matrix[:, [0]] == 1)
    for i in range(2, len(tran_log_index)):
        log_matrix[tran_log_index[i], 10] = log_matrix[tran_log_index[i],
                                                       2] - log_matrix[tran_log_index[i-1], 3]
    return log_matrix

# *************************************************************************************************************************************************
# *************************************************************************************************************************************************


tran_detect_debug()
#tran_log(res_trans, v_t, f_sys, sample_per_cycle)
