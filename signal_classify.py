import scipy.io
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import pandas as pd

#  List to contain all signal objects
signalList = []
# dictionary of all signals scores and labels
full_set = {}
# list of labels of all signals
labels = []
# list of scores of all signals
scores = []


# class signal from which signal objects are instantiated
class Signal():
    # signal TXID
    signal_id = 0
    # signal type (either GPS L1 or SBAS L1)
    signal_type = 1
    # scenario (clean,ds2,ds3,ds7)
    signal_scenario = ''
    # carrier to noise ratio
    CN = []
    # is signal spoofed or not.. used to build labels list
    spoofed = False
    # in band received power
    in_band_power = []
    # receiver clock error
    nav_Sol = []

    def __init__(self, channelfilename,
                 powerfilename,
                 navsolfilename,
                 navsolname,
                 powername,
                 channelname,
                 signaltype,
                 signalid,
                 spoofed=False):
        """
        Parameters
        ----------
        channelfilename: str
            absolute path to signal's channel.mat file
        powerfilename: str
            absolute path to signal's power.mat file
        navsolfilename: str
            absolute path to signal's navsol.mat file
        navsolname: str
            name of navsolxxxx.mat  matlab variable for specific scenario
        powername:str
             name of powerxxxx.mat  matlab variable for specific scenario
        channelname: str
            name of channelxxxx.mat matlab variable for specific scenario
        signaltype: int
            type of signal (0 for GPS L1 signals, 13 for SBAS l1 signals)
        signalid: int
            TXID of signal
        spoofed: bool
            weather signal is spoofed or not
        """
        # adding each signal constructed into signals list
        signalList.append(self)
        # loading matlab variables into python objects and initialising object attributes
        channel = scipy.io.loadmat(channelfilename).get(channelname)
        self.in_band_power = scipy.io.loadmat(powerfilename).get(powername)
        self.nav_sol = scipy.io.loadmat(navsolfilename).get(navsolname)
        self.CN = (channel[:, channel[13] == signalid])
        self.spoofed = spoofed
        self.signal_number = signalid
        self.signal_type = signaltype

    def get_signal_carrier_noise_ratio_diff(self):
        """
            returns max-min observed carrier to noise ratio

        Returns
        -------
            float

        """
        # return self.data[8, :2278]
        return max(self.CN[8, :2278]) - min(self.CN[8, :2278])

    def get_in_band_received_power_diff(self):
        """
            returns max-min observed in band received power

        Returns
        -------
            float

        """
        return max(self.in_band_power[0, :]) - min(self.in_band_power[0, :])

    def get_receiver_clock_error_diff(self):
        """
            returns max-min observed receiver clock error

        Returns
        -------
            float

        """
        return max(self.nav_sol[6, :]) - min(self.nav_sol[6, :])


def round_half_up(n, decimals=0):
    """
        rounds a float n to given decimal place decimals

    Parameters
    ----------
    n: float
        the float we want to round

    decimals: int
        decimal places to which we would like to round


    Returns
    -------
        float

    """
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier

# instantiating 56 signals (14 signal for each scenario)
signal3clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean',
           'navsolclean', 'power8clean', 'channelclean', 0, 3))
signal3ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 3, True))
signal3ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 3, True))
signal3ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 3, True))
signal6clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 0, 6))
signal6ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 6, True))
signal6ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 6, True))
signal6ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 6, True))
signal7clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 0, 7))
signal7ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 7, True))
signal7ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 7, True))
signal7ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 7, True))
signal10clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 0, 10))
signal10s2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 10, True))
signal10ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 10, True))
signal10ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 10, True))
signal11clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 0, 11))
signal11ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 11, True))
signal11ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 11, True))
signal11ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 11, True))
signal13clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 0, 13))
signal13ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 13, True))
signal13ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 13, True))
signal13ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 13, True))
signal16clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 0, 16))
signal16ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 16, True))
signal16ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 16, True))
signal16ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 16, True))
signal19clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 0, 19))
signal19ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 19, True))
signal19ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 19, True))
signal19ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 19, True))
signal20clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 0, 20))
signal20ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 20, True))
signal20ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 20, True))
signal20ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 20, True))
signal23clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 0, 23))
signal23ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 23, True))
signal23ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 23, True))
signal23ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 23, True))
signal30clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean',
           'navsolclean', 'power8clean', 'channelclean', 0, 30))
signal30ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 30, True))
signal30ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 30, True))
signal30ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 30, True))

signal32clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 0, 32))
signal32ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 0, 32, True))
signal32ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 0, 32, True))
signal32ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 0, 32, True))
signal133clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 13, 133))
signal133ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 13, 133, True))
signal133ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 13, 133, True))
signal133ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 13, 133, True))
signal138clean = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_clean.mat', r'C:\Users\intel\OneDrive\Desktop\powerclean',
           r'C:\Users\intel\OneDrive\Desktop\navsolclean', 'navsolclean',
           'power8clean', 'channelclean', 13, 138))
signal138ds2 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds2.mat', r'C:\Users\intel\OneDrive\Desktop\powerds2',
           r'C:\Users\intel\OneDrive\Desktop\navsolds2', 'navsolds2', 'power8ds2',
           'channelds2', 13, 138, True))
signal138ds3 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds3.mat', r'C:\Users\intel\OneDrive\Desktop\powerds3',
           r'C:\Users\intel\OneDrive\Desktop\navsolds3', 'navsolds3', 'power8ds3',
           'channelds3', 13, 138, True))
signal138ds7 = (
    Signal(r'C:\Users\intel\OneDrive\Desktop\channel_ds7.mat', r'C:\Users\intel\OneDrive\Desktop\powerds7',
           r'C:\Users\intel\OneDrive\Desktop\navsolds7', 'navsolds7', 'power8ds7',
           'channelds7', 13, 138, True))
# iterating through the signals list and building our features and labels dictionary
for signal in signalList:
    scores.append(abs(round_half_up(signal.get_in_band_received_power_diff(), 0) *
                      round_half_up(
                          signal.get_signal_carrier_noise_ratio_diff()) * 6.75 - signal.get_receiver_clock_error_diff()))
    labels.append((signal.spoofed))

full_set['scores'] = scores
full_set['labels'] = labels
# performing K means clustering
colmap = {1: 'r', 2: 'g', 3: 'b'}
kmeans = KMeans(n_clusters=2)
kmeans.fit(pd.DataFrame(full_set))
labels = kmeans.predict(pd.DataFrame(full_set))
centroids = kmeans.cluster_centers_
fig = plt.figure(figsize=(5, 5))
colors = map(lambda x: colmap[x + 1], labels)
plt.scatter(full_set['scores'], full_set['labels'], alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx + 1])
# plotting the results of the classification
plt.show()
