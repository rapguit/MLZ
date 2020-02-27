"""
.. module:: data
.. moduleauthor:: Matias Carrasco Kind
"""
__author__ = 'Matias Carrasco Kind'

import numpy as np
import random
import copy
import sys

from astropy.io import fits as pf

from . import utils_mlz

def read_catalog(filename, myrank=0, check='no', get_ng='no', L_1=0, L_2=-1, A_T=''):
    """
    Read the catalog, either for training or testing
    currently accepting ascii tables, numpy tables

    .. todo::
        HDF5 files

    :param str filename: Filename of the catalod
    :param int myrank: current processor id, for parallel reading (not implemented)
    :param str check: To check the code, only uses 200 lines of catalog
    :param str get_ng: Just get the total number og galaxies in the catalog
    :param int L_1: if passed get catalog between L_1 and L_2
    :param int L_2: if passed get catalog between L_1 and L_2
    :return: The whole catalog
    :rtype: float array
    """
    if filename[-3:] == 'npy':
        filein = np.load(filename)
        if check == 'yes': filein = filein[0:200]
        if get_ng == 'yes': return len(filein)
        if L_2 != -1: filein = filein[L_1:L_2]
    elif filename[-4:] == 'fits':
        GH = pf.open(filename, mode='readonly', memmap=False)
        if get_ng == 'yes':
            if check == 'yes':
                GH.close()
                return 200
            ngt = GH[1].header['NAXIS2']  # it assumed is present, TODO: check automatically
            GH.close()
            return ngt
        if L_2 > -1:
            Ta = GH[1].data[L_1:L_2]
        else:
            if check == 'no':
                Ta = GH[1].data
            else:
                Ta = GH[1].data[0:200]
        if A_T != '':
            col_index = []
            klist = []
            for k in A_T.keys():
                if A_T[k]['ind'] >= 0:
                    col_index.append(A_T[k]['ind'])
                    klist.append(k)
                if A_T[k]['eind'] >= 0:
                    col_index.append(A_T[k]['eind'])
            filein = np.zeros((len(Ta), max(col_index) + 1))
            for k in klist:
                T_temp = Ta.field(k)
                filein[:, A_T[k]['ind']] = T_temp
                if A_T[k]['eind'] >= 0:
                    T_temp = Ta.field('e' + k)
                    filein[:, A_T[k]['eind']] = T_temp
        else:
            filein = np.array(Ta.tolist())
        GH.close()
        del Ta, T_temp
        try:
            del GH[1].data
        except:
            pass
    elif filename[-3:] == 'csv':
        filein = np.loadtxt(filename, delimiter=',')
        if check == 'yes': filein = filein[0:200]
        if get_ng == 'yes': return len(filein)
        if L_2 != -1: filein = filein[L_1:L_2]
    else:
        filein = np.loadtxt(filename)
        if check == 'yes': filein = filein[0:200]
        if get_ng == 'yes': return len(filein)
        if L_2 != -1: filein = filein[L_1:L_2]
    return filein


def create_random_realizations(AT, F, N, keyatt):
    """
    Create random realizations using error in magnitudes,
    saves a temporarily file on train data directory.
    Uses normal distribution

    .. todo::
        Add other distributions

    :param dict AT: dictionary with columns names and colum index
    :param float F: Training data
    :param int N: Number of realizations
    :param str keyatt: Attribute name to be predicted or classifed
    :return: Returns  an array with random realizations
    """
    BigCat = {}
    total = len(F)
    for key in AT.keys():
        if (key != keyatt): BigCat[key] = np.zeros((total, N))
    for i in range(total):
        for k in BigCat.keys():
            sigg = F[i][AT[k]['eind']]
            sigg = max(0.001, sigg)
            sigg = min(sigg, 0.3)
            if AT[k]['eind'] == -1: sigg = 0.00005
            BigCat[k][i] = np.random.normal(F[i][AT[k]['ind']], sigg, N)
    return BigCat


def make_AT(cols, attributes, keyatt):
    """
    Creates dictionary used on all routines

     .. note::

        Make sure all columns have different names, and error columns are the same as attribute
        columns with a 'e' in front of it, ex. 'mag_u' and 'emag_u'

    :param str cols: str array with column names from file
    :param str attributes:  attributes to be used from those columns
    :param str keyatt: Attribute to be predicted or classified
    :return: dictionary, each key correspond to an attribute and itself a dictionary  where 'ind' is the
        column index and 'eind' is the error column for the same attribute,
        ex., A={u:{'ind'=1, 'eind'=6}}
    :rtype: dict
    """
    AT = {}
    for nc in attributes:
        w = np.where(cols == nc)
        AT[nc] = {'type': 'real'}
    AT[keyatt] = {'type': 'real'}
    for c in AT.keys():
        j = np.where(cols == c)[0]
        ej = np.where(cols == 'e' + c)[0]
        if len(ej) == 0: ej = np.array([-1])
        if len(j) == 0: j = np.array([-1])
        AT[c]['ind'] = j[0]
        AT[c]['eind'] = ej[0]
    return AT


def bootstrap_index(N, SS):
    """
    Returns bootstrapping indexes of sample N from array of indices

    :param int N: size of boostrap sample
    :param int SS: extract indexes from 0 to SS
    :return: array of bootstrap indices
    :rtype: int array
    """
    index = []
    for i in range(N):
        index.append(random.randint(0, SS - 1))
    return np.array(index)
    # return stat.randint.rvs(0,SS,size=N)


class catalog():
    """
    Creates a catalog instance for training or testing

    :param class Pars: Class of parameters read from inputs files
    :param str cat_type: 'train' or 'test' file (names are taken from Pars class)
    :param int L1: keep only entries between L1 and L2
    :param int L2: keep only entries between L1 and L2
    """

    def __init__(self, Pars, cat_type='train', L1=0, L2=-1, rank=0):
        self.Pars = Pars
        self.cat_type = cat_type
        if cat_type == 'train':
            self.filename = Pars.path_train + Pars.trainfile
            self.cols = np.array(Pars.columns)
            if not Pars.keyatt in Pars.columns:
                if rank == 0: utils_mlz.printpz_err("Column ", Pars.keyatt,
                                                    " not found in training file, check inputs file")
                sys.exit(0)
        if cat_type == 'test':
            self.filename = Pars.path_test + Pars.testfile
            self.cols = np.array(Pars.columns_test)
        self.atts = Pars.att
        self.AT = make_AT(self.cols, self.atts, Pars.keyatt)
        self.cat = read_catalog(self.filename, check=Pars.checkonly, get_ng='no', L_1=L1, L_2=L2, A_T=self.AT)
        # if L2 != -1: self.cat = self.cat[L1:L2]
        self.cat_or = copy.deepcopy(self.cat)
        self.nobj = len(self.cat)
        self.ndim = len(self.atts)
        self.has_random = False
        self.oob = 'no'

    def has_X(self):
        """
        Is X already loaded in memory?

        :return: Boolean
        """
        try:
            type(self.X)
            return True
        except AttributeError:
            return False

    def has_Y(self):
        """
        Is Y already loaded in memory?

        :return: Boolean
        """
        try:
            type(self.Y)
            return True
        except AttributeError:
            return False

    def get_XY(self, curr_at='all', bootstrap='no'):
        """
        Creates X and Y methods based on catalog, using random realization or bootstrapping,
        after this both X and Y are loaded and ready to be used

        :param dict curr_at: dictionary  of attributes to be used (like a subsample of them), 'all' by default
        :param str bootstrap: Bootstrapping sample? ('yes'/'no')
        :return: Saves X, Y oob (and no-oob) data if required and original catalog
        """
        self.boot = bootstrap
        if curr_at == 'all':
            self.curr_at = self.atts
        else:
            self.curr_at = curr_at['atts']
        indx = []
        for key in self.curr_at:
            indx.append(self.AT[key]['ind'])
        indx = np.array(indx)
        self.indx = indx
        self.X = self.cat[:, indx]
        nboot = len(self.X)
        if self.oob == 'yes': self.Xoob = self.cat_oob[:, indx]
        if self.boot == 'yes':
            self.in_boot = bootstrap_index(nboot, nboot)
            self.X = self.X[self.in_boot]
        if self.boot == 'no':
            self.in_boot = np.arange(nboot)
        if self.AT[self.Pars.keyatt]['ind'] != -1:
            self.Y = self.cat[:, self.AT[self.Pars.keyatt]['ind']]
            if self.oob == 'yes': self.Yoob = self.cat_oob[:, self.AT[self.Pars.keyatt]['ind']]
            if self.boot == 'yes':
                self.Y = self.Y[self.in_boot]
        self.cat2 = copy.deepcopy(self.cat[self.in_boot])

    def make_random(self, outfileran='', ntimes=-1):
        """
        Actually makes the random realizations
        :param str outfileran: output file (not needed)
        :param int ntimes: taken from class Pars unless otherwise indicated
        """
        if ntimes == -1: ntimes = int(self.Pars.nrandom)
        if outfileran == '': outfileran = self.Pars.randomcatname
        self.BigRan = create_random_realizations(self.AT, self.cat, ntimes, self.Pars.keyatt)
        np.save(self.Pars.path_train + outfileran, self.BigRan)
        self.has_random = True

    def load_random(self):
        """
        Loads the random catalog with the realizations
        """
        Junk = np.load(self.Pars.path_train + self.Pars.randomcatname + '.npy', allow_pickle=True)
        self.BigRan = Junk.item()
        del Junk

    def newcat(self, i):
        self.cat = copy.deepcopy(self.cat_or)
        if i > 0:
            for k in self.AT.keys():
                if k != self.Pars.keyatt: 
                    self.cat[:, self.AT[k]['ind']] = self.BigRan[k][:, i]

    def oob_data(self, frac=0.):
        """
        Creates oob data and separates it from the no-oob data for further tests
        :param float frac: Fraction of the data to be separated, taken from class Pars (default is 1/3)
        """
        if not self.has_X() or not self.has_Y(): print('ERROR2')
        if frac == 0.: frac = self.Pars.oobfraction
        self.noob = int(self.nobj * frac)
        self.oob_index = random.sample(range(self.nobj), self.noob)
        index_all = np.arange(self.nobj)
        index_all[self.oob_index] = -1
        woob = np.where(index_all >= 0)[0]
        self.no_oob_index = index_all[woob]
        self.Xoob = self.X[self.oob_index]
        self.Yoob = self.Y[self.oob_index]
        self.X = self.X[self.no_oob_index]
        self.Y = self.Y[self.no_oob_index]
        self.oob_index_or = self.in_boot[self.oob_index]

    def oob_data_cat(self, frac=0.):
        self.oob = 'yes'
        self.cat = copy.deepcopy(self.cat_or)
        if frac == 0.: frac = self.Pars.oobfraction
        self.noob = int(self.nobj * frac)
        self.oob_index = random.sample(range(self.nobj), self.noob)
        index_all = np.arange(self.nobj)
        index_all[self.oob_index] = -1
        woob = np.where(index_all >= 0)[0]
        self.no_oob_index = index_all[woob]
        self.cat_oob = self.cat[self.oob_index]
        self.cat = self.cat[self.no_oob_index]
        self.oob_index_or = self.oob_index

    def sample_dim(self, nsample):
        """
        Samples from the list of attributes

        :param int nsample: size of subsample
        :return: dictionary with subsample attributes and their locations
        """
        self.ndim_sample = nsample
        r_dim = random.sample(self.atts, nsample)
        self.dict_dim = {}
        self.dict_dim['atts'] = r_dim
        for k in r_dim:
            self.dict_dim[k] = self.AT[k]['ind']
        return self.dict_dim

