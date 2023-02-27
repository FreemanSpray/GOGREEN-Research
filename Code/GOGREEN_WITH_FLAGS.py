from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
import scipy.optimize as opt



class GOGREEN:
    def __init__(self, dataPath:str):
        """
        __init__ Constructor to define and initialize class members

        :param dataPath: absolute path to the directory containing DR1/ and STRUCTURAL_PARA_v1.1_CATONLY/
                         subdirectories
        """ 
        
        self.catalog = pd.DataFrame()
        self.sourceCatalog = pd.DataFrame()
        self.standardCriteria = []
        # Private Members
        self._path = dataPath
        self._structClusterNames = ['SpARCS0219', 'SpARCS0035','SpARCS1634', 'SpARCS1616', 'SPT0546', 'SpARCS1638',
                                    'SPT0205', 'SPT2106', 'SpARCS1051', 'SpARCS0335', 'SpARCS1034']
        self._clustersCatalog = pd.DataFrame()
        self._photoCatalog = pd.DataFrame()
        self._redshiftCatalog = pd.DataFrame()
        self._galfitCatalog = pd.DataFrame()
        self._matchedCatalog = pd.DataFrame()
        self._photSourceCatalog = pd.DataFrame()
        self._specSourceCatalog = pd.DataFrame()

        self.init()
    # END __INIT__

    def init(self):
        """
        init Helper method for initializing catalogs
        """ 
        # Build path string to the cluster catalog
        clusterCatPath = self._path + 'DR1/CATS/Clusters.fits'
        # Generate a DataFrame of the catalog data
        self._clustersCatalog = self.generateDF(clusterCatPath)
        # Remove whitespaces included with some cluster names
        self._clustersCatalog['cluster'] = self._clustersCatalog['cluster'].str.strip()
        """
        # Build path string to the photometric catalog
        photoCatPath = self._path + 'DR1/CATS/Photo.fits'
        # Generate a DataFrame of the catalog data
        self._photoCatalog = self.generateDF(photoCatPath)
        """
        # Build path string to the redshift catalog
        redshiftCatPath = self._path + 'DR1/CATS/Redshift_catalogue.fits'
        # Generate a DataFrame of the catalog data
        self._redshiftCatalog = self.generateDF(redshiftCatPath)

        # Build path string to the phot source catalogue
        photSourceCatPath = self._path + 'STELLPOPS_V2/photometry_stellpops.fits'
        # Generate a DataFrame of the catalog data
        self._photSourceCatalog = self.generateDF(photSourceCatPath)

        # Build path string to the spec source catalogue
        specSourceCatPath = self._path + 'STELLPOPS_V2/redshifts_stellpops.fits'
        # Generate a DataFrame of the catalog data
        self._specSourceCatalog = self.generateDF(specSourceCatPath)

        # Merge source catalogs
        merge_col = ['SPECID']
        # This only ouputs columns with names different than those in the redshift table.  
        # Make sure that SPECID is added back in as we will match on that field
        cols_to_use = self._specSourceCatalog.columns.difference(self._photSourceCatalog.columns).tolist() + merge_col
        self._photoCatalog = self.merge(self._photSourceCatalog, self._specSourceCatalog[cols_to_use], merge_col)

        # Build a DataFrame for each galfit and matched structural parameter cluster (11 total)
        # Then combine them into a single galfit catalog and a single matched catalog
        galfitCatPath = self._path + 'STRUCTURAL_PARA_v1.1_CATONLY/GALFIT_ORG_CATS/'
        matchedCatPath = self._path + 'STRUCTURAL_PARA_v1.1_CATONLY/STRUCTCAT_MATCHED/'
        for clusterName in self._structClusterNames:
            # Build filename strings
            galfitClusterFilename = 'gal_' + clusterName.lower() + '_orgcat.fits'
            matchedClusterFilename = 'structcat_photmatch_' + clusterName.lower() + '.dat'
            # Filename for SpARCSXXXX clusters in photmatched catalogs are spjXXXX
            if (clusterName[:6] == 'SpARCS'):
                galfitClusterFilename = 'gal_spj' + clusterName[-4:] + '_orgcat.fits'
                matchedClusterFilename = 'structcat_photmatch_spj' + clusterName[-4:] + '.dat'

            # Generate a DataFrame of the galfit cluster data
            galfitClusterDF = self.generateDF(galfitCatPath + galfitClusterFilename)
            # Combine it with the main galfit DataFrame
            self._galfitCatalog = self._galfitCatalog.append(galfitClusterDF)

            # Generate a DataFrame of the struct matched cluster data
            matchedClusterDF = self.generateDF(matchedCatPath + matchedClusterFilename)
            # Convert PHOTCATID to cPHOTID
            # Find a cPHOTID of the cluster in the photometric catalog 
            tempCPHOTID = self._photoCatalog[self._photoCatalog['Cluster'] == clusterName].iloc[0]['cPHOTID']
            # Extract the source ID and cluster ID from the temporary cPHOTID
            idPrefix = int(str(tempCPHOTID)[:3])*int(1e6)
            # Convert the structural catalog PHOTCATID into the photometric catalog cPHOTID
            matchedClusterDF.rename(columns = {'PHOTCATID':'cPHOTID'}, inplace = True)
            matchedClusterDF.loc[:,'cPHOTID'] += idPrefix
            # Combine it with the main struct matched DataFrame
            self._matchedCatalog = self._matchedCatalog.append(matchedClusterDF)

        # Merge photomatched structural catalog with photometric catalog
        self.catalog = self.merge(self._photoCatalog, self._matchedCatalog, 'cPHOTID')
        # readjust to preserve NaN values after merge
        self.catalog = self.catalog.replace(100000000000000000000, np.nan)
        # Generate flags for use in plotting
        self.generateFlags()
        # Set error values (necessary because the re_err values from Galfit are not adequate)
        self.setReErr()
        # Generate unit conversion fields for use in effective radius plots.
        self.reConvert()
        self.catalog['re_frac_err'] = self.catalog['re_err_robust']/self.catalog['re']
        self.catalog['re_frac_err_converted'] = self.catalog['re_err_robust_converted']/self.catalog['re_converted']

    # END INIT

    def generateFlags(self):
        # Initialize queries
        #passiveQuery = '(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)' #(from van der Burg et al. 2020)
        passiveQuery = 'NUVMINV > 2 * VMINJ + 1.6' #(𝑁𝑈𝑉 − 𝑉 ) > 2(𝑉 − 𝐽) + 1.6(from McNab et al 2021)
        #starFormingQuery = '(UMINV <= 1.3) or (VMINJ >= 1.6) or (UMINV <= 0.60+VMINJ)' #(from van der Burg et al. 2020)
        starFormingQuery = 'NUVMINV < 2 * VMINJ + 1.1' #(𝑁𝑈𝑉 − 𝑉 ) < 2(𝑉 − 𝐽) + 1.1 (from McNab et al 2021)
        gvQuery = '(2 * VMINJ + 1.1 <= NUVMINV) and (NUVMINV <= 2 * VMINJ + 1.6)' # 2(𝑉 − 𝐽) + 1.1 ≤ (𝑁𝑈𝑉 − 𝑉 ) ≤ 2(𝑉 − 𝐽) + 1.6 (from McNab et al 2021)
        bqQuery = '((VMINJ + 0.45 <= UMINV) and (UMINV <= VMINJ + 1.35)) and ((-1.25 * VMINJ + 2.025 <= UMINV) and (UMINV <= -1.25 * VMINJ + 2.7))' # (𝑉 − 𝐽) + 0.45 ≤ (𝑈 − 𝑉 ) ≤ (𝑉 − 𝐽) + 1.35 ### − 1.25 (𝑉 − 𝐽) + 2.025 ≤ (𝑈 − 𝑉 ) ≤ −1.25 (𝑉 − 𝐽) + 2.7 (from McNab et al 2021)
        psbQuery = 'D4000 < 1.45 and delta_BIC < -10' # (D4000 < 1.45) ∩ (ΔBIC < −10) (from McNab et al 2021)
        ellipticalQuery = '2.5 < n < 6'
        spiralQuery = 'n <= 2.5'
        # Initialize flags
        self.catalog['goodData'] = 1
        reduced = self.catalog[~self.catalog['zspec'].isna()]
        self.catalog['spectroscopic'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)
        reduced = self.catalog[self.catalog['zspec'].isna()]
        self.catalog['photometric'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int) # this is in line with how phot was being calculated previously. We may want to try a solution not dependent on the spec calculation
        reduced = self.catalog.query(passiveQuery)
        self.catalog['passive'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)
        reduced = self.catalog.query(starFormingQuery)
        self.catalog['starForming'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)
        reduced = self.catalog.query(gvQuery)
        self.catalog['greenValley'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)
        reduced = self.catalog.query(bqQuery)
        self.catalog['blueQuiescent'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)
        reduced = self.catalog.query(psbQuery)
        self.catalog['postStarBurst'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)
        reduced = self.catalog.query(ellipticalQuery)
        self.catalog['elliptical'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)
        reduced = self.catalog.query(spiralQuery)
        self.catalog['spiral'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)
        self.catalog['member_adjusted'] = 0
        for clusterName in self._structClusterNames:
            reduced = self.getMembers(clusterName, 2)
            self.catalog['member_adjusted'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int) + self.catalog['member_adjusted'] 
    # END GENERATEFLAGS

    def generateDF(self, filePath:str) -> pd.DataFrame:
        """
        generateDF Generates a Pandas DataFrame from a .fits or .dat file

        :param filePath: Relative path from the directory containing DR1/ and STRUCTURAL_PARA_v1.1_CATONLY/
                         subdirectories to the desired data file to load into a DataFrame
        :return:         Pandas DataFrame containing the data stored in param:filePath
        """
        # Different data formats require different DataFrame initializations
        # Extract the data file format from the file path
        fileType = os.path.splitext(filePath)[1]
        if fileType == '.fits':
            return Table( fits.getdata( filePath ) ).to_pandas()
        elif fileType == '.dat':
            if 'STRUCTURAL_PARA_v1.1_CATONLY' in filePath:
                headers = pd.read_csv(filePath, delim_whitespace=True, nrows=0, skiprows=1).columns[1:]
                return pd.read_csv(filePath, sep='\s+', engine='python', header=None, skiprows=2, names=headers)
            return pd.read_csv(filePath, sep='\s+', engine='python', header=1)
        else:
            print("The ", fileType, " data format is not current implemented!")
            return pd.DataFrame()
    # END GENERATEDF

    def merge(self, frame1:pd.DataFrame, frame2:pd.DataFrame, columnName:str=None) -> pd.DataFrame:
        """
        merge Combines two Pandas DataFrames along the axis specified by param:columnName
              Only values of param:columnName contained in both frames will be kept

        :param frame1:     Pandas DataFrame to combine with param:frame2
        :param frame2:     Pandas DataFrame to combine with param:frame1
        :param columnName: Name of column shared between param:frame1 and param:frame2 to join on
                            Default: None
        :return:           Pandas DataFrame containing param:frame1 and param:frame2 merged on the param:columnName axis
        """
        return pd.merge(frame1, frame2, how='left', on=columnName)
    # END MERGE

    def getClusterZ(self, clusterName:str) -> float:
        """
        getClusterZ Gets the best estimate of the cluster redshift for the cluster specified by param:clusterName

        :param clusterName: Name of the cluster whose redshift should be returned
        :return:            Cluster redshift estimate as a float
        """
        targetCluster = self._clustersCatalog[self._clustersCatalog['cluster'] == clusterName]
        return targetCluster['Redshift'].values[0]
    # END GETCLUSTERZ

    def getMembers(self, clusterName:str, offset:float=1) -> pd.DataFrame:
        """
        getMembers Gets the member galaxies of a cluster based on the galaxy redshift with respect to the
                   best estimate of the cluster redshift. Spectroscopic members are those with (zspec-zclust) < 0.02(1+zspec)
                   and photometric members are those with (zphot-zclust) < 0.08(1+zphot).

        :param clusterName: Name of the cluster whose members should be returned
        :return:            Pandas DataFrame containing the galaxies whose redshift match the membership requirements
        """
        clusterZ = self.getClusterZ(clusterName)
        allClusterGalaxies = self.getClusterGalaxies(clusterName)
        # Find spectroscopic and photometric members seperately
        # Spectrosocpic criteria: (zspec-zclust) < 0.02(1+zspec)
        specZthreshold = np.abs(allClusterGalaxies['zspec'].values-clusterZ) <= offset*0.02*(1+allClusterGalaxies['zspec'].values)
        specZgalaxies = allClusterGalaxies[specZthreshold]
        # Photometric criteria: (zphot-zclust) < 0.08(1+zphot)
        photZthreshold = np.abs(allClusterGalaxies['zphot'].values-clusterZ) <= offset*0.08*(1+allClusterGalaxies['zphot'].values)
        photZgalaxies = allClusterGalaxies[photZthreshold]
        # Remove photZgalaxies with a specZ
        photZgalaxies = photZgalaxies[~photZgalaxies['cPHOTID'].isin(specZgalaxies['cPHOTID'])]
        # Combine into a single DataFrame
        memberGalaxies = specZgalaxies.append(photZgalaxies)
        return memberGalaxies
    # END GETMEMBERS

    def getNonMembers(self, clusterName:str) -> pd.DataFrame:
        """
        getNonMembers Gets the non-member galaxies (field galaxies in the line of sight of the cluster, either in front of or behind it) 
                   of a cluster based on the galaxy redshift with respect to the best estimate of the cluster redshift. 
                   Spectroscopic members are those with (zspec-zclust) < 0.02(1+zspec) and photometric members are those with (zphot-zclust) < 0.08(1+zphot). 
                   Thus the non-members will be the opposite.

        :param clusterName: Name of the cluster whose non-members should be returned
        :return:            Pandas DataFrame containing the galaxies whose redshift match the membership requirements
        """
        allClusterGalaxies = self.getClusterGalaxies(clusterName)
        memberGalaxies = self.getMembers(clusterName)
        nonMemberGalaxies = allClusterGalaxies[~allClusterGalaxies['cPHOTID'].isin(memberGalaxies['cPHOTID'])]
        return nonMemberGalaxies
    # END GETNONMEMBERS

    def reduceDF(self, additionalCriteria:list, useStandards:bool) -> pd.DataFrame:
        """
        reduceDF Reduces the DataFrame param:frame to contain only galaxies that meet the criteria provided in
                 param:additionalCriteria and the standard criteria (if param:useStandards is True)

        :param additionalCriteria: List of criteria to apply to param:frame
        :param useStandards:       Flag to specify whether the standard criteria should be applied to param:frame
        :return:                   Pandas DataFrame containing the galaxies whose values meet the criteria within param:additionalCriteria
                                   and the standard criteria (if param:useStandards is True)
        """
        # Reinitialize flag
        self.catalog['goodData'] = 1
        if (additionalCriteria != None):
            for criteria in additionalCriteria:
                reduced = self.catalog.query(criteria)
                self.catalog['goodData'] = self.catalog['goodData'] & (self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int))
        if useStandards:
            for criteria in self.standardCriteria:
                reduced = self.catalog.query(criteria)
                self.catalog['goodData'] = self.catalog['goodData'] & (self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)) # https://www.geeksforgeeks.org/python-pandas-series-astype-to-convert-data-type-of-series/
        return self.catalog
    # END REDUCEDF
        
    def getClusterGalaxies(self, clusterName:str) -> pd.DataFrame:
        """
        getClusterGalaxies Get all galaxies associated with the cluster provided by param:clusterName

        :param clusterName: Name of the cluster whose galaxies should be returned
        :return:            Pandas DataFrame containing only galaxies associated with cluster param:clusterName 
        """

        return self.catalog[self.catalog['Cluster'] == clusterName]
    # END GETCLUSTERGALAXIES

    def convertData(self, source:pd.DataFrame, target:pd.DataFrame) -> pd.DataFrame:
        ret = target[target['cPHOTID'].isin(source['cPHOTID'])]
        return ret
    # END convertData

    def plotPassiveLines(self, axes:list=None, row:int=None, col:int=None):
        """
        plotPassiveLines (private method) draws the recognized boundary between passive and star-forming galaxies on UVJ plots
        :param axes:                The array of subplots created when the plotType is set to 2.
                                     Default: None
        :param row :                Specifies the row of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :param col :                Specifies the column of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :return    :                lines are plotted
        """

        # Generate the data used to plot the line
        x = [-5, 0.7, 1.6, 1.6]
        y = [1.3, 1.3, 2.2, 5]
        # In case of subplots, plot for the specific row and column
        if row != None and col != None:
            if axes[row][col] != None:
                axes[row][col].plot(x, y, color='black')
                return
        # Else plot normally
        plt.plot(x, y, color='black')
    #END PLOTPASSIVELINES

    def plotVanDerWelLines(self, axes:list=None, row:int=None, col:int=None):
        """
        plotVanDerWelLines plots the MSR line calculated in van der Wel et al. 2014

        :param axes:                The array of subplots created when the plotType is set to 2.
                                     Default: None
        :param row :                Specifies the row of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :param col :                Specifies the column of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :return    :                lines are plotted
        """
        
        # Generate the data used to plot the line
        Asf = pow(10, 0.7)
        Apassive = pow(10, 0.22)
        alphaSF = 0.22
        alphaPassive = 0.76
        xVals = np.array([9.5, 11.5])
        MstellarRange = pow(10, xVals)
        yValsPassive = np.log10(np.array([Apassive * pow((i / (5 * np.float_power(10, 10))), alphaPassive) for i in MstellarRange]))
        yValsSF = np.log10(np.array([Asf * pow((i / (5 * np.float_power(10, 10))), alphaSF) for i in MstellarRange]))
        # In case of subplots, plot for the specific row and column
        if row != None and col != None:
            if axes[row][col] != None:
                axes[row][col].plot(xVals, yValsPassive, linestyle='dashed', color='red')
                axes[row][col].plot(xVals, yValsSF, linestyle='dashed', color='blue')
                return
        # Else plot normally
        plt.plot(xVals, yValsPassive, linestyle='dashed', color='red')
        plt.plot(xVals, yValsSF, linestyle='dashed', color='blue')
    #END PLOTVANDERWELLINES

    def plotMcNabPlots(self):
        """
        plotMcNabPlots plots two plots from McNab et. al. 2021

        :return    :     plots are plotted
        """
        # Reduce according to criteria
        self.reduceDF(None, True)
        print(self.catalog.query('goodData == 1').shape[0])
        # Reduce to set of good data
        table = pd.DataFrame()
        table['Population'] = ['SF', 'Q', 'GV', 'BQ', 'PSB']
        table['Total Sample'] = [self.catalog.query('goodData == 1 and starForming == 1').shape[0], 
            self.catalog.query('goodData == 1 and passive == 1').shape[0], 
            self.catalog.query('goodData == 1 and greenValley == 1').shape[0], 
            self.catalog.query('goodData == 1 and blueQuiescent == 1').shape[0], 
            self.catalog.query('goodData == 1 and postStarBurst == 1').shape[0]]
        table['Cluster Members'] = [self.catalog.query('goodData == 1 and starForming == 1').shape[0], 
            self.catalog.query('goodData == 1 and passive == 1').shape[0], 
            self.catalog.query('goodData == 1 and greenValley == 1').shape[0], 
            self.catalog.query('goodData == 1 and blueQuiescent == 1').shape[0], 
            self.catalog.query('goodData == 1 and postStarBurst == 1').shape[0]]
        print(table)
        # Extract desired quantities from data
        passiveMembersBad = self.catalog.query('goodData == 1 and passive == 1 and Mstellar <= 1.6e10')
        starFormingMembersBad = self.catalog.query('goodData == 1 and starForming == 1 and Mstellar <= 1.6e10')
        greenValleyMembersBad = self.catalog.query('goodData == 1 and greenValley == 1 and Mstellar <= 1.6e10')
        blueQuiescentMembersBad = self.catalog.query('goodData == 1 and blueQuiescent == 1 and Mstellar <= 1.6e10')
        postStarBurstMembersBad = self.catalog.query('goodData == 1 and postStarBurst == 1 and Mstellar <= 1.6e10')

        passiveMembersGood = self.catalog.query('goodData == 1 and passive == 1 and Mstellar > 1.6e10')
        starFormingMembersGood = self.catalog.query('goodData == 1 and starForming == 1 and Mstellar > 1.6e10')
        greenValleyMembersGood = self.catalog.query('goodData == 1 and greenValley == 1 and Mstellar > 1.6e10')
        blueQuiescentMembersGood = self.catalog.query('goodData == 1 and blueQuiescent == 1 and Mstellar > 1.6e10')
        postStarBurstMembersGood = self.catalog.query('goodData == 1 and postStarBurst == 1 and Mstellar > 1.6e10')

        plt.figure()
        plt.scatter(passiveMembersBad['VMINJ'], passiveMembersBad['NUVMINV'], alpha=0.5, s=15, marker='o', color='red')
        plt.scatter(starFormingMembersBad['VMINJ'], starFormingMembersBad['NUVMINV'], alpha=0.5, s=15, marker='*',  color='blue')
        plt.scatter(greenValleyMembersBad['VMINJ'], greenValleyMembersBad['NUVMINV'], alpha=0.5, s=15, marker='d', color='green')
        plt.scatter(blueQuiescentMembersBad['VMINJ'], blueQuiescentMembersBad['NUVMINV'], alpha=0.5, s=15, marker='s', color='orange')
        plt.scatter(postStarBurstMembersBad['VMINJ'], postStarBurstMembersBad['NUVMINV'], alpha=0.5, s=15, marker='x', color='purple')
        plt.scatter(passiveMembersGood['VMINJ'], passiveMembersGood['NUVMINV'], alpha=0.5, s=60, marker='o', color='red')
        plt.scatter(starFormingMembersGood['VMINJ'], starFormingMembersGood['NUVMINV'], alpha=0.5, s=60, marker='*',  color='blue')
        plt.scatter(greenValleyMembersGood['VMINJ'], greenValleyMembersGood['NUVMINV'], alpha=0.5, s=60, marker='d', color='green')
        plt.scatter(blueQuiescentMembersGood['VMINJ'], blueQuiescentMembersGood['NUVMINV'], alpha=0.5, s=60, marker='s', color='orange', label='BQ')
        plt.scatter(postStarBurstMembersGood['VMINJ'], postStarBurstMembersGood['NUVMINV'], alpha=0.5, s=60, marker='x', color='purple', label='PSB')
        plt.plot([0.2, 2], [2, 5.5], linestyle='dashed', color='black')
        plt.plot([0.2, 2], [1.5, 5], linestyle='dashed', color='black')
        plt.fill_between([0.2, 2], [1.5, 5], [2, 5.5], color='green', alpha=0.1)
        plt.xlabel("(V-J)")
        plt.ylabel("(NUV-V)")
        plt.xlim(0.2, 2)
        plt.ylim(1, 6)
        plt.legend()

        plt.figure()
        plt.scatter(passiveMembersBad['VMINJ'], passiveMembersBad['UMINV'], alpha=0.5, s=15, marker='o', color='red')
        plt.scatter(starFormingMembersBad['VMINJ'], starFormingMembersBad['UMINV'], alpha=0.5, s=15, marker='*',  color='blue')
        plt.scatter(greenValleyMembersBad['VMINJ'], greenValleyMembersBad['UMINV'], alpha=0.5, s=15, marker='d', color='green')
        plt.scatter(blueQuiescentMembersBad['VMINJ'], blueQuiescentMembersBad['UMINV'], alpha=0.5, s=15, marker='s', color='orange')
        plt.scatter(postStarBurstMembersBad['VMINJ'], postStarBurstMembersBad['UMINV'], alpha=0.5, s=15, marker='x', color='purple')
        plt.scatter(passiveMembersGood['VMINJ'], passiveMembersGood['UMINV'], alpha=0.5, s=30, marker='o', color='red', label='Q')
        plt.scatter(starFormingMembersGood['VMINJ'], starFormingMembersGood['UMINV'], alpha=0.5, s=60, marker='*',  color='blue', label='SF')
        plt.scatter(greenValleyMembersGood['VMINJ'], greenValleyMembersGood['UMINV'], alpha=0.5, s=60, marker='d', color='green', label='GV')
        plt.scatter(blueQuiescentMembersGood['VMINJ'], blueQuiescentMembersGood['UMINV'], alpha=0.5, s=60, marker='s', color='orange')
        plt.scatter(postStarBurstMembersGood['VMINJ'], postStarBurstMembersGood['UMINV'], alpha=0.5, s=60, marker='x', color='purple')
        #xPoints = [0.3, 0.6, 0.7, 1]
        #yPoints = [1.65, 1.95, 1.2, 1.45]
        plt.plot([0.3, 0.6], [1.65, 1.95], linestyle='dashed', color='black') # top left
        plt.plot([0.7, 1], [1.2, 1.45], linestyle='dashed', color='black') # bottom right
        plt.plot([0.6, 1], [1.95, 1.45], linestyle='dashed', color='black') # top right
        plt.plot([0.3, 0.7], [1.65, 1.2], linestyle='dashed', color='black') # bottom left
        plt.fill_between([0.3, 0.6], [1.65, 1.3], [1.65, 1.95], color='orange', alpha=0.1)
        plt.fill_between([0.7, 1], [1.2, 1.45], [1.85, 1.45], color='orange', alpha=0.1)
        plt.fill_between([0.6, 0.7], [1.3, 1.2], [1.95, 1.85], color='orange', alpha=0.1)
        plt.xlabel("(V-J)")
        plt.ylabel("(U-V)")
        plt.xlim(0.25, 2.25)
        plt.ylim(0.5, 2.6)
        plt.legend()
    #END PLOTMCNABPLOTS

    def setReErr(self):
        """
        Assign error values for Effective Radius (Re) measurements. These values are taken from van der Wel et. al. 2012, Table 3
        :return   :    values are set in the catalog
        """
        self.catalog['re_err_robust'] = np.nan
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 21) & (self.catalog.re < 0.3), 0.01, self.catalog.re_err_robust) #https://stackoverflow.com/questions/12307099/modifying-a-subset-of-rows-in-a-pandas-dataframe
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 21) & (self.catalog.re > 0.3), 0.00, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 22) & (self.catalog.re < 0.3), 0.02, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 22) & (self.catalog.re > 0.3), -0.01, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 23) & (self.catalog.re < 0.3), 0.00, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 23) & (self.catalog.re > 0.3), -0.03, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 24) & (self.catalog.re < 0.3), 0.01, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 24) & (self.catalog.re > 0.3), -0.10, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 25) & (self.catalog.re < 0.3), 0.04, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 25) & (self.catalog.re > 0.3), -0.09, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 26) & (self.catalog.re < 0.3), 0.12, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 26) & (self.catalog.re > 0.3), -0.11, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 27) & (self.catalog.re < 0.3), 0.27, self.catalog.re_err_robust)
        
    #END SETREERR

    def reConvert(self):
        """
        reConvert convert effective radius values from units of arcsec to kpc.

        :param data:   The set of data being used by the calling function, plot().
        :return   :    returns the list of converted effective radius values

        """
        sizes =  self.catalog['re'].values
        errs = self.catalog['re_err_robust'].values
        length = len(sizes)
        # Convert all effective radii from units of arcsec to kpc using their spectroscopic redshifts
        sizes_converted = sizes * (cosmo.kpc_proper_per_arcmin(self.catalog['zspec'].values)/60)
        errs_converted = errs * (cosmo.kpc_proper_per_arcmin(self.catalog['zspec'].values)/60)
        # Try zphot values where conversion failed due to lack of zspec value
        for i in range(0, length):
            if np.isnan(sizes_converted[i]):
                sizes_converted[i] = sizes[i] * (cosmo.kpc_proper_per_arcmin(self.catalog['zphot'].values[i])/60)
            if np.isnan(errs_converted[i]):
               errs_converted[i] = errs[i] * (cosmo.kpc_proper_per_arcmin(self.catalog['zphot'].values[i])/60)
        # Remove units
        sizes_converted = (sizes_converted / u.kpc) * u.arcmin
        errs_converted = (errs_converted / u.kpc) * u.arcmin
        self.catalog['re_converted'] = sizes_converted
        self.catalog['re_err_robust_converted'] = errs_converted
    #END RECONVERT

    def MSRfit(self, data:list, useLog:list=[False, False], axes:list=None, row:int=None, col:int=None, typeRestrict:str=None, color:str=None, bootstrap:bool=True):
        """
        MSRfit fits a best fit line to data generated by the plot() method

        :param data:                The set of data that is relevant to the plot already generated by the plot() method
        :param useLog:              Flag to indicate whether the x- or y-axis should be in log scale
                                     Default:   [False,False] - neither axis in log scale
                                     Value:   [False,True] - y axis in log scale
                                     Value:   [True,False] - x axis in log scale
                                     Value:   [True,True] - both axis in log scale
        :param axes:                The array of subplots created when the plotType is set to 2.
                                     Default: None
        :param row:                 Specifies the row of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :param col:                 Specifies the column of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :param typeRestrict:        Flag to indicate whether data should be restricted based on SFR (only necessary when allData is True)
                                     Default: None
                                     Value:   'Quiescent' - only passive galaxies should be considered.
                                     Value:   'Star-Forming' - only star forming galaxies should be considered.
                                     Value:   'Elliptical' - only galaxies with 2.5 < n < 6 should be considered.
                                     Value:   'Spiral' - only galaxies with n < 2.5 should be considered.
        :param color1:             The color the fit line should be.
                                    Default: 'black'     
        :param bootstrap           Flag to indicate rather bootstrapping should be used to calculate and display uncertainty on the fit  
        :return   :

        """
        # Establish label
        if typeRestrict == None:
            lbl = "stellar mass-size relation trend"
        else:
            lbl = typeRestrict + " stellar mass-size relation trend"
        # Extract values frmo data
        size = data['re_converted'].values
        mass = data['Mstellar'].values
        errs = data['re_err_robust_converted'].values
        # Calculate coefficients (slope and y-intercept)
        if useLog[0] == True:
            mass = np.log10(mass)
        if useLog[1] == True:
            upperErrs = np.log10(size + errs) - np.log10(size)
            lowerErrs = np.log10(size) - np.log10(size - errs)
            size = np.log10(size)
            errs = (upperErrs + lowerErrs)/2
        # Transform error values into weights
        weights = 1/np.array(errs)
        for i in range(0, len(weights)): # Explanation of the error that provoked this check: https://predictdb.org/post/2021/07/23/error-linalgerror-svd-did-not-converge/
            if np.isinf(weights[i]):
                weights[i] = 0 #setting to 0 because this data point should not be used
            if np.isnan(weights[i]):
                weights[i] = 0 #setting to 0 because this data point should not be used  
        s = np.polynomial.polynomial.Polynomial.fit(x=mass, y=size, deg=1, w=weights)
        coefs = s.convert().coef
        intercept = coefs[0]
        slope = coefs[1]
        print((slope, intercept))
        for i in range(0, len(errs)): # Explanation of the error that provoked this check: https://predictdb.org/post/2021/07/23/error-linalgerror-svd-did-not-converge/
            #errs[i] = 1 # NOTE: temp, for testing/display purposes
            if np.isinf(errs[i]):
                errs[i] = 10000 #setting to arbitrarily high because this data point should not be used
            if np.isnan(errs[i]):
                errs[i] = 10000 #setting to arbitrarily high because this data point should not be used
        print(errs)  
        # Note: we define bounds here because this causes the default fitting method to be changed to trf, which in 
        # turn causes the function to call scipy.optimize.least_squares internally, which can take the loss param
        #s, _ = opt.curve_fit(f=lambda x, m, b: m*x + b, xdata=mass, ydata=size, p0=[slope, intercept], sigma=errs, bounds=([-10, -10], [10, 10]), loss="soft_l1")
        #s, _ = opt.curve_fit(f=lambda x, m, b: m*x + b, xdata=mass, ydata=size, sigma=errs)
        guessVals = [slope, intercept]
        s, _ = opt.curve_fit(f=lambda x, m, b: m*x + b, xdata=mass, ydata=size, p0=guessVals, bounds=([-10, -10], [10, 10]), loss="huber")
        slope = s[0]
        intercept = s[1]
        if row != None and col != None:
            # Check for subplots
            if axes[row][col] != None:
                if bootstrap:
                    # Bootstrapping calculation
                    #self.bootstrap(mass, size, weights, axes, row, col, lineColor=color, guessVals=guessVals)
                    self.bootstrap(mass, size, errs, axes, row, col, lineColor=color, guessVals=guessVals)
                # Add white backline in case of plotting multiple fit lines in one plot
                if color != 'black':
                    axes[row][col].plot(mass, intercept + slope*mass, color='white', linewidth=4)
                # Plot the best fit line
                axes[row][col].plot(mass, intercept + slope*mass, color=color, label=lbl)
                return
        if bootstrap:
            # Bootstrapping calculation
            #self.bootstrap(mass, size, weights, axes, row, col, lineColor=color, guessVals=guessVals)
            self.bootstrap(mass, size, errs, axes, row, col, lineColor=color, guessVals=guessVals)
        # Add white backline in case of plotting multiple fit lines in one plot
        if color != 'black':
            plt.plot(mass, intercept + slope*mass, color='white', linewidth=4)
        # Plot the best fit line
        plt.plot(mass, intercept + slope*mass, color=color, label=lbl)
        return slope, intercept
    # END MSRFIT

    def bootstrap(self, x:list=None, y:list=None, error:list=None, axes:list=None, row:int=None, col:int=None, lineColor:str=None, guessVals:list=None):
        """
        bootstrap obtains a measure of error of the line-fitting equation ...
        
        :param x:                   List containing the mass values of the data set
                                     Default: None
        :param y:                   List containing the size values corresponding to each mass value in the data set
                                     Default: None
        :param error:               List containing the error values corresponding to each size value in the data set
                                     Default: None
        :param axes:                The array of subplots created when the plotType is set to 2.
                                     Default: None
        :param row:                 Specifies the row of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :param col:                 Specifies the column of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :param lineColor:           Flag to indicate what color should be used to accentuate the trendline.
                                     Default: None
        :return      :    ...
        """
        # Establish type of plot
        plot = plt
        # Check for subplots
        if row != None and col != None:
            # Check for subplots
            if axes[row][col] != None:
                plot = axes[row][col]
        # Initialize seed for consistent results across runs
        rng = np.random.RandomState(1234567890) # reference: https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array
        # Initialize arrays
        size = len(x)
        xMin = np.min(x)
        xMax = np.max(x)
        slopes = np.empty((100,))
        intercepts = np.empty((100,))
        # Create 100 bootstrap lines
        for i in range(0, 100):
            plotted = False
            while not(plotted):
                # Initialize new array of synthetic data
                randIndices = rng.randint(0, size, size=size)
                # Fill mutatedX with randomly selected mass values from x
                bootstrapX = x[randIndices]
                bootstrapY = y[randIndices]
                boostrapE = error[randIndices]
                # Fit data with equation
                try:
                    #s = np.polynomial.polynomial.Polynomial.fit(x=bootstrapX, y=bootstrapY, deg=1, w=boostrapE)
                    s, _ = opt.curve_fit(f=lambda x, m, b: m*x + b, xdata=bootstrapX, ydata=bootstrapY, p0=guessVals, sigma=boostrapE, bounds=([-10, -10], [10, 10]), loss="huber")
                    #s, _ = opt.curve_fit(f=lambda x, m, b: m*x + b, xdata=bootstrapX, ydata=bootstrapY, p0=guessVals, bounds=([-10, -10], [10, 10]), loss="huber")
                    m = s[0]
                    b = s[1]
                    # Store coefficients
                    intercepts[i] = b
                    slopes[i] = m
                    # Calculate outputs
                    xline = np.array([xMin, xMax])
                    yline = b + m*xline # Equivalent operation: yline = s(xline)
                    # Plot curve
                    #plot.plot(xline, yline, color='green', alpha=0.6)
                    plotted = True
                except RuntimeError:
                    print("caught runtime error")
                except np.linalg.LinAlgError:
                    print("caught linear algebra error")
        # Create grid of points to test calculated m & b values at.
        xGrid = np.linspace(xMin, xMax, 1000)
        gridSize = len(xGrid)
        yGrid = np.empty((gridSize, 100))
        # Initialize interval endpoint storage
        yTops = np.empty((gridSize,))
        yBots = np.empty((gridSize,))
        for i in range(0, gridSize):
            for j in range(0, 100):
                # Calculate y using y = mx + b for the ith grid coordinate for the jth bootstrap line
                yGrid[i][j] = intercepts[j] + xGrid[i]*slopes[j]
        for i in range(0, gridSize):
            # Calculate 68% confidence interval for the ith grid coordinate
            yTops[i] = np.percentile(yGrid[i], 84)
            yBots[i] = np.percentile(yGrid[i], 16)
        # Determine color to be used
        # Useful tool: https://www.rapidtables.com/web/color/RGB_Color.html
        if lineColor == 'red':
            color = [0.5, 0, 0] # darker red
        elif lineColor == 'green':
            color = [0, 0.5, 0] # darker green
        elif lineColor == 'orange':
            color = [1, 0.8, 0.8] # pinkish
        # star-forming and default case
        else:
            color = [0, 0, 0.5] # darker blue
        # Plot curves on top and bottom of intervals
        plot.plot(xGrid, yTops, color=color)
        plot.plot(xGrid, yBots, color=color)
        plot.fill_between(xGrid, yBots, yTops, color=color, alpha=0.5) # https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html
    # END BOOTSTRAP

    def getRatio(self, x:float=None, y:float=None, bootstrap:bool=True, limitRange:bool=True, useTransition:bool=False) -> list:
        """
        Calculates the ratio of member over non-member galaxies (for both passive and star-forming). Also has an option to plot the trendlines for these 4 categories

        :param category:    Name of the category to consider when making comparisons
                             Default: 'SF' - indicates passive vs star-forming should be compared
        :param x:           X value at which the comparison should be made
                             Default: None
        :param y:           Y value at which the comparison should be made
                             Default: None
        :param plotLines:  
        :param xRange:     List containing the desired lower and upper bounds for the x-axis
                            Default: None
        :param yRange:     List containing the desired lower and upper bounds for the y-axis
                            Default: None
        :return: size 2 list of the two ratios of member over non-member galaxies (first element is for quiescent, second is for star-forming)
        """
        # Adjust plot size
        plt.figure(figsize=(10,10))
        # Reduce according to criteria
        self.reduceDF(None, True)
        # Initialize dataframes            
        members = pd.DataFrame()
        nonMembers = pd.DataFrame()
        # For each cluster, add members and non-members to respective dataframes
        for clusterName in self._structClusterNames:
            members = members.append(self.getMembers(clusterName))
            nonMembers = nonMembers.append(self.getNonMembers(clusterName))
        # Extract desired quantities from data
        passiveMembers = members.query('passive == 1 and goodData == 1')
        passiveNonMembers = nonMembers.query('passive == 1 and goodData == 1')
        starFormingMembers = members.query('starForming == 1 and goodData == 1')
        starFormingNonMembers = nonMembers.query('starForming == 1 and goodData == 1')
        # Plot quiescent and sf trends for members and nonmembers (4 lines total)
        mMemberQ, bMemberQ = self.MSRfit(data=passiveMembers, useLog=[True, True], typeRestrict='Quiescent cluster', color='red', bootstrap=bootstrap)
        mMemberSF, bMemberSF = self.MSRfit(data=starFormingMembers, useLog=[True, True], typeRestrict='Star-Forming cluster', color='blue', bootstrap=bootstrap)
        mNonMemberQ, bNonMemberQ = self.MSRfit(data=passiveNonMembers, useLog=[True, True], typeRestrict='Quiescent field', color='orange', bootstrap=bootstrap)
        mNonMemberSF, bNonMemberSF = self.MSRfit(data=starFormingNonMembers, useLog=[True, True], typeRestrict='Star-Forming field', color='green', bootstrap=bootstrap)
        # Transition galaxy option
        if useTransition:
            # Extracted desired quantities from data
            gvMembers = members.query('greenValley == 1 and goodData == 1')
            gvNonMembers = nonMembers.query('greenValley == 1 and goodData == 1')
            bqMembers = members.query('blueQuiescent == 1 and goodData == 1')
            bqNonMembers = nonMembers.query('blueQuiescent == 1 and goodData == 1')
            psbMembers = members.query('postStarBurst == 1 and goodData == 1')
            psbNonMembers = nonMembers.query('postStarBurst == 1 and goodData == 1')
            # Plot trends (6 additional lines)
            _, _ = self.MSRfit(data=gvMembers, useLog=[True, True], typeRestrict='GV cluster', color='purple', bootstrap=bootstrap)
            _, _ = self.MSRfit(data=gvNonMembers, useLog=[True, True], typeRestrict='GV field', color='pink', bootstrap=bootstrap)
            _, _ = self.MSRfit(data=bqMembers, useLog=[True, True], typeRestrict='BQ cluster', color='black', bootstrap=bootstrap)
            _, _ = self.MSRfit(data=bqNonMembers, useLog=[True, True], typeRestrict='BQ field', color='gray', bootstrap=bootstrap)
            _, _ = self.MSRfit(data=psbMembers, useLog=[True, True], typeRestrict='PSB cluster', color='brown', bootstrap=bootstrap)
            #_, _ = self.MSRfit(data=psbNonMembers, useLog=[True, True], typeRestrict='PSB field', color='yellow', bootstrap=bootstrap)
        if limitRange:
            plt.xlim(9.5, 11.5)
            plt.ylim(-0.75, 1.25)
        plt.legend()
        if x != None:
            # Get ratios at a certain x value
            pointMemberQ = x*mMemberQ + bMemberQ
            pointMemberSF = x*mMemberSF + bMemberSF 
            pointNonMemberQ = x*mNonMemberQ + bNonMemberQ 
            pointNonMemberSF = x*mNonMemberSF + bNonMemberSF
            ratioQ = pointMemberQ/pointNonMemberQ
            ratioSF = pointMemberSF/pointNonMemberSF
            return [ratioQ, ratioSF]
        elif y != None:
            # Get ratios at a certain y value
            pointMemberQ = (y/mMemberQ) - (bMemberQ/mMemberQ)
            pointMemberSF = (y/mMemberSF) - (bMemberSF/mMemberSF)
            pointNonMemberQ = (y/mNonMemberQ) - (bNonMemberQ/mNonMemberQ) 
            pointNonMemberSF = (y/mNonMemberSF) - (bNonMemberSF/mNonMemberSF) 
            ratioQ = pointMemberQ/pointNonMemberQ
            ratioSF = pointMemberSF/pointNonMemberSF
            return [ratioQ, ratioSF]
        else:
            print("No point of comparison provided")
            return [-1]     
    #END GETRATIO

    def getMedian(self, category:str='SF', xRange:list=None, yRange:list=None):
        """
        Plots the median in four mass bins, including uncertainty and standard error on the median

        :param category  :     Name of the category to consider when making comparisons
        :param xRange    :     List containing the desired lower and upper bounds for the x-axis
                                Default: None
        :param yRange    :     List containing the desired lower and upper bounds for the y-axis
                                Default: None
        :return: medians and uncertainties are plotted
        """
        # Set an initial value to append to.
        memberData = pd.DataFrame()
        nonMemberData = pd.DataFrame()
        for clusterName  in self._structClusterNames:
            # Get data for this cluster for galaxies classified as members
            memberData = memberData.append(self.getMembers(clusterName))
            nonMemberData = nonMemberData.append(self.getNonMembers(clusterName))
        # Apply other specified reducing constraints
        memberData = self.reduceDF(memberData, None, True)
        nonMemberData = self.reduceDF(nonMemberData, None, True)
        # Handle case where only passive galaxies out of all data need to plotted.
        if category == 'SF':
            memberDataQ = memberData.query('(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)')
            memberDataSF = memberData.query('(UMINV <= 1.3) or (VMINJ >= 1.6) or (UMINV <= 0.60+VMINJ)')
            nonMemberDataQ = nonMemberData.query('(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)')
            nonMemberDataSF = nonMemberData.query('(UMINV <= 1.3) or (VMINJ >= 1.6) or (UMINV <= 0.60+VMINJ)')
            # Separate each of the 4 data sets into 4 mass bins
            MQbins = np.array([memberDataQ.query('(Mstellar > 3162280000) and (Mstellar < 10000000000)'), memberDataQ.query('Mstellar > 10000000000 and Mstellar < 31622800000'), memberDataQ.query('Mstellar > 31622800000 and Mstellar < 100000000000'), memberDataQ.query('Mstellar > 100000000000 and Mstellar < 316228000000')])
            MSFbins = np.array([memberDataSF.query('(Mstellar > 3162280000) and (Mstellar < 10000000000)'), memberDataSF.query('Mstellar > 10000000000 and Mstellar < 31622800000'), memberDataSF.query('Mstellar > 31622800000 and Mstellar < 100000000000'), memberDataSF.query('Mstellar > 100000000000 and Mstellar < 316228000000')])
            NMQbins = np.array([nonMemberDataQ.query('(Mstellar > 3162280000) and (Mstellar < 10000000000)'), nonMemberDataQ.query('Mstellar > 10000000000 and Mstellar < 31622800000'), nonMemberDataQ.query('Mstellar > 31622800000 and Mstellar < 100000000000'), nonMemberDataQ.query('Mstellar > 100000000000 and Mstellar < 316228000000')])
            NMSFbins = np.array([nonMemberDataSF.query('(Mstellar > 3162280000) and (Mstellar < 10000000000)'), nonMemberDataSF.query('Mstellar > 10000000000 and Mstellar < 31622800000'), nonMemberDataSF.query('Mstellar > 31622800000 and Mstellar < 100000000000'), nonMemberDataSF.query('Mstellar > 100000000000 and Mstellar < 316228000000')])
            # Convert all effective radii from units of arcsec to kpc using their spectroscopic redshifts, and convert to log.
            sizeMQ = np.array([np.log10(self.reConvert(i)) for i in MQbins])
            sizeMSF = np.array([np.log10(self.reConvert(i)) for i in MSFbins])
            sizeNMQ= np.array([np.log10(self.reConvert(i)) for i in NMQbins])
            sizeNMSF = np.array([np.log10(self.reConvert(i)) for i in NMSFbins])
            # Calculate median for each data set
            medianMQ = np.array([np.median(i) for i in sizeMQ])
            medianMSF = np.array([np.median(i) for i in sizeMSF])
            medianNMQ = np.array([np.median(i) for i in sizeNMQ])
            medianNMSF = np.array([np.median(i) for i in sizeNMSF])
            # Use 4 mass bins for x values
            xValues = np.array([9.75, 10.25, 10.75, 11.25])
            # Create offsets to make uncertainties legible
            offsetMQ = 0
            offsetMSF = 0.03
            offsetNMQ = -0.03
            offsetNMSF = 0.06
            # Plot the median at each mass bin
            plt.scatter(xValues + offsetMQ, medianMQ, label='quiescent members')
            plt.scatter(xValues + offsetMSF, medianMSF, label='star-forming members')
            plt.scatter(xValues + offsetNMQ, medianNMQ, label='quiescent non-members')
            plt.scatter(xValues + offsetNMSF, medianNMSF, label='star-forming non-members')
            plt.legend()
            # Plot uncertainty bars
            for i in range(0, len(xValues)):
                self.plotUncertainties(sizeMQ[i], medianMQ[i], xValues[i] + offsetMQ, 'blue')
                self.plotUncertainties(sizeMSF[i], medianMSF[i], xValues[i] + offsetMSF, 'orange')
                self.plotUncertainties(sizeNMQ[i], medianNMQ[i], xValues[i] + offsetNMQ, 'green')
                self.plotUncertainties(sizeNMSF[i], medianNMSF[i], xValues[i] + offsetNMSF, 'red')
            # Limit range of the plot to be the xRange and yRange parameters
            if xRange != None:
                if len(xRange) > 1:
                    plt.xlim(xRange[0], xRange[1])
            if yRange != None:
                if len(yRange) > 1:
                    plt.ylim(yRange[0], yRange[1])
            # Label axes
            plt.xlabel('log(Mstellar)')
            plt.ylabel('log(Re)')
            # Calculate standard error for each of the 16 medians
            stdErrorsMQ = np.array([self.getStdError(i) for i in sizeMQ])
            stdErrorsMSF = np.array([self.getStdError(i) for i in sizeMSF])
            stdErrorsNMQ = np.array([self.getStdError(i) for i in sizeNMQ])
            stdErrorsNMSF = np.array([self.getStdError(i) for i in sizeNMSF])
            # Plot standard error bars
            for i in range(0, len(xValues)):
                self.plotStdError(medianMQ[i], xValues[i] + offsetMQ, stdErrorsMQ[i], 'black')
                self.plotStdError(medianMSF[i], xValues[i] + offsetMSF, stdErrorsMSF[i], 'black')
                self.plotStdError(medianNMQ[i], xValues[i] + offsetNMQ, stdErrorsNMQ[i], 'black')
                self.plotStdError(medianNMSF[i], xValues[i] + offsetNMSF, stdErrorsNMSF[i], 'black')
            # NOTE: size (dy) of upper and lower error bar will be asymetric because it is LOG
    #END GETMEDIAN

    def getStdError(self, data:list=None) -> int:
        """
        getStdError calculates standard error of the median of a list of sizes
        
        :param data    :     List containing the size values of the data set
                                Default: None
        :return: standard error value
        """
        return 1.253 * (np.std(data)/np.sqrt(len(data)))
    #END GETSTDERROR

    def plotStdError(self, median:int=None, bin:int=None, stdError:int=None, color:str=None):
        """
        plotStdError plots error bars for the standard error of the median
        
        :param median :     the median size value
                                Default: None
        :param bin :        the mass value corresponding to the median
                                Default: None
        :param stdError :   the standard error of the median
                                Default: None
        :param color    :   the color of the error bars
                                Default: None                        
        :return: error bars are plotted
        """
        plt.errorbar(bin, median, stdError, barsabove = True, ecolor=color)
        plt.errorbar(bin, median, stdError, barsabove = False, ecolor=color)
    #END PLOTSTDERROR

    def plotUncertainties(self, data:list=None, median:int=None, bin:int=None, color:str=None):
        """
        plotUncertainties plots error bars for the uncertainty on the median
        
        :param median :     the median size value
                                Default: None
        :param bin :        the mass value corresponding to the median
                                Default: None
        :param color    :   the color of the error bars
                                Default: None                        
        :return: error bars are plotted
        """
        confLower = np.percentile(data, 25)
        confHigher = np.percentile(data, 75)
        # Plot upper error (75 percent)
        plt.errorbar(bin, median, confHigher - median, barsabove=True, ecolor=color)
        # Plot lower error (25 percent)
        plt.errorbar(bin, median, median - confLower, barsabove=False, ecolor=color)
    #END PLOTUNCERTAINTY

    def makeTable(self, filename):
        """
        Generates a table of slope and intercept values of best fit lines of all, passive, and star forming galaxies in each cluster

        :param : filename - the name of the file to write to
        :return: writes slopes and y-intercepts to the file 'output.txt'
        """
        # Open file for writing
        f = open(filename, 'w')
        # Create headers
        f.write('# Cluster Slope_All Y-Intercept_All Slope_Passive Y-Intercept_Passive Slope_SF Y-Intercept_SF\n')
        # Create data sets
        data = pd.DataFrame()
        for clusterName  in self._structClusterNames:
            # Get all galaxies associated with this cluster
            # Reduce data to only contain galaxies classified as members
            data = self.getMembers(clusterName)
            # Apply standard reducing constraints
            data = self.reduceDF(data, None, True)
            # Create separate data frames of only passive and star forming galaxies.
            passive = data.query('(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)')
            starForming = data.query('(UMINV <= 1.3) or (VMINJ >= 1.6) or (UMINV <= 0.60+VMINJ)')
            
            # Write the cluster name
            f.write(clusterName + ' ')
            # Write the data (3 sets, each with 2 statistics)
            self.writeTable(data, f)
            self.writeTable(passive, f)
            self.writeTable(starForming, f)
            f.write('\n')

        # Close file
        f.close()
    # END MAKETABLE

    def writeTable(self, data, f):
        """
        Helper function for use by makeTable()
        
        :param : data - Pandas data frame correlating to one cluster. May be the entirety of the data for the cluster (after standard criteria have been applied)
                        May also be only the data for passive or star forming galaxies.
        :param : f - the file to write to (assumes file is open)
        :return: writes slope and intercept of best fit line for the data to output.txt.
        """
        # Convert all effective radii from units of arcsec to kpc using their spectroscopic redshifts
        size = self.reConvert(data)
        # Extract mass data
        mass = data['Mstellar'].values
        # Exclude data points that would cause an error
        mass, size = self.cutBadData(mass, size)
        
        xFitData = np.log10(mass)
        yFitData = np.log10(size)
        m, b = np.polyfit(xFitData, yFitData, 1) #slope and intercept for best fit line
        f.write(str(m) + ' ')
        f.write(str(b) + ' ')    
    # END MAKETABLE

    def testPlots(self):
        """
        Makes a series of test plots, stores and analyzes the data counts resulting from each plot to judge the accuracy of the plot() function.

        :post:      counts are written to file C:/Users/panda/Documents/Github/GOGREEN-Research/Notebooks/testOutput.txt
        """
        # Establish criteria
        searchCriteria = [
            'Star == 0',
            'K_flag == 0',
            'totmask == 0',
            're > 0',
            'Mstellar > 6300000000',
            'Fit_flag > 1',
            'n < 6',
            'HSTFOV_flag == 1',
            '(1 < zspec < 1.5) or ((Redshift_Quality != 3) and (Redshift_Quality != 4) and (1 < zphot < 1.5))'
        ]
        self.standardCriteria = searchCriteria

        with warnings.catch_warnings(): #suppressing depracation warnings for readability purposes
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
            # Open file for writing
            f = open('C:/Users/panda/Documents/Github/GOGREEN-Research/Notebooks/testOutput.txt', 'w')
            # Establish variables for first test
            memberStatus = ["all", "only", "not"]
            plotType = [1, 2, 3]
            colorType = [None, "membership", "passive", "sersic"]
            # Plot MSR and UVJ plots for each variable
            for m in memberStatus:
                for p in plotType:
                    for c in colorType:
                        if p == 1:
                            cluster = "SpARCS1616"
                        else:
                            cluster = None
                        xCountMSR, yCountMSR = self.plot('Mstellar', 're_converted', plotType=p, clusterName=cluster, useMembers=m, colorType=c, useLog=[True,True], xRange = [9.5, 11.5], yRange = [-1.5, 1.5], xLabel='log(Mstellar)', yLabel='log(Re)', fitLine=False)
                        xCountUVJ, yCountUVJ = self.plot('VMINJ', 'UMINV', plotType=p, clusterName=cluster, useMembers=m, colorType=c, useLog=[False,False], xRange = [-0.5,2.0], yRange = [0.0, 2.5], xLabel='V - J', yLabel='U - V', fitLine=False)
                        # End test early (and with specific error) if major discrepency is found
                        if xCountMSR != yCountMSR or xCountUVJ != yCountUVJ:
                            print("test failed. X and Y data counts do not agree.")
                            return
                        if xCountMSR != xCountUVJ:
                            print("test failed. MSR and UVJ counts do not agree.")
                            return
                        # Write data count (all four return values will be the same if this line is reached so we only need to write once)
                        f.write(str(xCountMSR) + ' ')
            # Establish variables for second test
            clusterNames = ["SpARCS0219", "SpARCS0035", "SpARCS1634", "SpARCS1616", "SPT0546", "SpARCS1638", "SPT0205", "SPT2106", "SpARCS1051", "SpARCS0335", "SpARCS1034"]
            xTot = 0
            # Seperate results with newline
            f.write('\n')
            # Plot MSR plot for each cluster
            for cluster in clusterNames:
                xCount, _ = self.plot('Mstellar', 're', plotType=1, clusterName=cluster, useMembers="only", colorType=c, useLog=[True,True], xRange = [9.5, 11.5], yRange = [-1.5, 1.5], xLabel='log(Mstellar)', yLabel='log(Re)', fitLine=False)
                # Write data count
                f.write(str(xCount) + ' ')
                # Add value to total
                xTot+=xCount
            # Write total count on another newline
            f.write('\n' + str(xTot) + ' ')
            # Plot MSR plot for all clusters combined
            xTotExpected, _ = self.plot('Mstellar', 're', plotType=3, clusterName=cluster, useMembers="only", colorType=c, useLog=[True,True], xRange = [9.5, 11.5], yRange = [-1.5, 1.5], xLabel='log(Mstellar)', yLabel='log(Re)', fitLine=False)
            # Write expected total
            f.write(str(xTotExpected))
            if xTot != xTotExpected:
                print("test failed. Totaled Individual and combined cluster counts do not agree.")
                return
            # Establish variables for third test
            clusterNames = ["SpARCS0219", "SpARCS0035", "SpARCS1634", "SpARCS1616", "SPT0546", "SpARCS1638", "SPT0205", "SPT2106", "SpARCS1051", "SpARCS0335", "SpARCS1034"]
            xTot = 0
            # Seperate results with newline
            f.write('\n')
            # Plot MSR plot for each cluster
            for cluster in clusterNames:
                xCount, _ = self.plot('Mstellar', 're', plotType=1, clusterName=cluster, useMembers="not", colorType=c, useLog=[True,True], xRange = [9.5, 11.5], yRange = [-1.5, 1.5], xLabel='log(Mstellar)', yLabel='log(Re)', fitLine=False)
                # Write data count
                f.write(str(xCount) + ' ')
                # Add value to total
                xTot+=xCount
            # Write total count on another newline
            f.write('\n' + str(xTot) + ' ')
            # Plot MSR plot for all clusters combined
            xTotExpected, _ = self.plot('Mstellar', 're', plotType=3, clusterName=cluster, useMembers="not", colorType=c, useLog=[True,True], xRange = [9.5, 11.5], yRange = [-1.5, 1.5], xLabel='log(Mstellar)', yLabel='log(Re)', fitLine=False)
            # Write expected total
            f.write(str(xTotExpected))
            if xTot != xTotExpected:
                print("test failed. Totaled Individual and combined cluster counts do not agree.")
                return
            f.close()
            f = open('C:/Users/panda/Documents/Github/GOGREEN-Research/Notebooks/testOutput.txt', 'r')
            testOutput = f.read()
            f.close()
            f = open('C:/Users/panda/Documents/Github/GOGREEN-Research/Notebooks/truth.txt', 'r')
            expectedOutput = f.read()
            f.close()
            if testOutput == expectedOutput:
                print("test passed.")
                return
            print("test failed due to unspecified error case.")
    # END TEST
    
    def plotUnwrapped(self, xQuantityName:str, yQuantityName:str, colorType:str=None, useLog:list=[False,False], fitLine:bool=False,
        data:pd.DataFrame=None, color1:list=None, color2:list=None, plot=None, axes:list=None, row:int=None, col:int=None, bootstrap:bool=True, plotErrBars:bool=False,):
            """
            Helper function called by plot. Handles the plotting of data.
                
            :param xQuantityName:      Name of the column whose values are to be used as the x
            :param yQuantityName:      Name of the column whose values are to be used as the y
            :param colorType:          Specifies how to color code the plotted galaxies
                                        Default: None
                                        Value:   'membership' - spectroscopic member vs photometric member
                                        Value:   'passive' - passive vs star forming
            :param useLog:             Flag to indicate whether the x- or y-axis should be in log scale
                                        Default: [False,False] - neither axis in log scale
                                        Value:   [False,True] - y axis in log scale
                                        Value:   [True,False] - x axis in log scale
                                        Value:   [True,True] - both axis in log scale
            :param fitLine:            Flag to indicate whether a best fit line should be fit to the data. By default this line will plot size vs mass. 
                                        (note: the default x and y will be in log, however specifically selected values will correspond to the useLog list)
                                        (note: not currently configured to work with plot type 2)
            :param data:               Set of data points to be plotted. 
            :param color1:             Specifies what color should be used when plotting first type of data
                                        Value:   (r,g,b)
            :param color2:             Specifies what color should be used when plotting first type of data
                                        (note: unused when colorType is None)
                                        Value:   (r,g,b)
            :param plot:               The plot on which the data should be plotted.
                                        Value: will be either the module 'plt' or a subplot
            :param axes:               The array of subplots created when the plotType is set to 2.
                                        Default: None
            :param row:                Specifies the row of the 2D array of subplots. For use when axes is not None.
                                        Default: None
            :param col:                Specifies the column of the 2D array of subplots. For use when axes is not None.
                                        Default: None
            :param holdLbls:           Indicates whether or not labels should be plotted. Needed when plotting multiple times on the same plot.
            :post:                     The generated plot(s) will be displayed
            :return:                   (x, y), representing the total number of x-values and y-values corresponding to plotted data points
            """
            # Arbitrary establishment of variables for non-coloring case
            aData = data.query('goodData == 1')
            bData = data.query('goodData == 1')
            aLbl = None
            bLbl = None
            # Overwrite variables according to coloring scheme
            if colorType == None:
                # Don't need to do anything for this case. Included so program proceeds as normal
                pass
            elif colorType == 'membership':
                aData = data.query('spectroscopic == 1 and goodData == 1')
                aLbl = 'Spectroscopic z'
                bData = data.query('photometric == 1 and goodData == 1')
                bLbl = 'Photometric z'
            elif colorType == 'passive':
                aData = data.query('passive == 1 and goodData == 1')
                aLbl = 'Quiescent'
                bData = data.query('starForming == 1 and goodData == 1')
                bLbl = 'Star Forming'
            elif colorType == 'GV':
                aData = data.query('greenValley == 1 and goodData == 1')
                aLbl = 'Green Valley'
                bData = data.query('greenValley == 0 and goodData == 1')
                bLbl = 'Other'
            elif colorType == 'BQ':
                aData = data.query('blueQuiescent == 1 and goodData == 1')
                aLbl = 'Blue Quiescent'
                bData = data.query('blueQuiescent == 0 and goodData == 1')
                bLbl = 'Other'
            elif colorType == 'PSB': 
                aData = data.query('postStarBurst == 1 and goodData == 1')
                aLbl = 'Post-starburst'
                bData = data.query('postStarBurst == 0 and goodData == 1')
                bLbl = 'Other' 
            elif colorType == 'sersic':
                aData = data.query('elliptical == 1 and goodData == 1')
                aLbl = 'Elliptical'
                bData = data.query('spiral == 1 and goodData == 1')
                bLbl = 'Spiral'
            else:
                print(colorType, ' is not a valid coloring scheme!')
                return
            aXVals = aData[xQuantityName].values
            bXVals = bData[xQuantityName].values
            aYVals = aData[yQuantityName].values
            bYVals = bData[yQuantityName].values
            # Check if either axis needs to be put in log scale
            if useLog[0] == True:
                aXVals = np.log10(aXVals)
                bXVals = np.log10(bXVals)
            if useLog[1] == True:
                aYVals = np.log10(aYVals)
                bYVals = np.log10(bYVals)
            # Plot passive v star-forming border in the case where we are plotting UVJ color-color
            if xQuantityName == 'VMINJ' and yQuantityName == 'UMINV':
                self.plotPassiveLines(axes, row, col)
            # generate best fit line
            if fitLine:
                # Generate two if plotting two distinct categories
                if colorType != None:
                    self.MSRfit(aData, useLog, axes, row, col, typeRestrict=aLbl, color=color1, bootstrap=bootstrap)
                    self.MSRfit(bData, useLog, axes, row, col, color=color2, typeRestrict=bLbl, bootstrap=bootstrap)
                else:
                    #print(aData.shape)
                    self.MSRfit(aData, useLog, axes, row, col, bootstrap=bootstrap)
            # Generate the plot
            plot.scatter(aXVals, aYVals, alpha=0.5, color=color1, label=aLbl)
            if plotErrBars:
                # Extract error values
                if yQuantityName == 're':
                    aYsigmas = aData['re_err_robust'].values
                    bYsigmas = bData['re_err_robust'].values
                elif yQuantityName == 're_converted':
                    aYsigmas = aData['re_err_robust_converted'].values
                    bYsigmas = bData['re_err_robust_converted'].values
                for i in range(0, len(aXVals)):
                    mass = aXVals[i]
                    size = aYVals[i]
                    sigma = aYsigmas[i]
                    upperSigma = np.log10(pow(10, size) + sigma) - np.log10(pow(10, size))
                    lowerSigma = np.log10(pow(10, size)) - np.log10(pow(10, size) - sigma)
                    if np.isnan(upperSigma) or np.isnan(lowerSigma):
                        plt.scatter(mass, size, alpha=0.5, color='black')
                    else:
                        plt.errorbar(mass, size, upperSigma, barsabove = True, ecolor='red')
                        plt.errorbar(mass, size, lowerSigma, barsabove = False, ecolor='red')
                for i in range(0, len(bXVals)):
                    mass = bXVals[i]
                    size = bYVals[i]
                    sigma = bYsigmas[i]
                    upperSigma = np.log10(size + sigma) - np.log10(size)
                    lowerSigma = np.log10(size) - np.log10(size - sigma)
                    if np.isnan(upperSigma) or np.isnan(lowerSigma):
                        plt.scatter(mass, size, alpha=0.5, color='black')
                    else:
                        plt.errorbar(mass, size, upperSigma, barsabove = True, ecolor='blue')
                        plt.errorbar(mass, size, lowerSigma, barsabove = False, ecolor='blue')
            if colorType != None:
                plot.scatter(bXVals, bYVals, alpha=0.5, color=color2, label=bLbl)
            # Plot van der Wel et al. 2014 line in the case where we are plotting MSR (passive v starforming)
            if xQuantityName == 'Mstellar' and (yQuantityName == 're' or yQuantityName == 're_converted') and colorType == "Passive":
                self.plotVanDerWelLines()
            # Return data counts (used when running test suite)
            xA = aXVals.shape[0]
            yA = aYVals.shape[0]
            if colorType != None:
                xB = bXVals.shape[0]
                yB = bYVals.shape[0]
            else:
                xB = 0
                yB = 0
            return (xA + xB, yA + yB)
    # END PLOTUNWRAPPED


    def plot(self, xQuantityName:str, yQuantityName:str, plotType:int, clusterName:str=None, additionalCriteria:list=None, useMembers:str='only', colorType:str=None, colors:list=None, 
        useStandards:bool=True, xRange:list=None, yRange:list=None, xLabel:str='', yLabel:str='', useLog:list=[False,False], fitLine:bool=False, bootstrap:bool=True, plotErrBars:bool=False):
        """
        plot Generates a plot(s) of param:xQuantityName vs param:yQuantityName according to param:plotType
             
        :param xQuantityName:      Name of the column whose values are to be used as the x
        :param yQuantityName:      Name of the column whose values are to be used as the y
        :param plotType:           How to plots should be generated
                                    Value: 1 - plot only the cluster provided in param:clusterName
                                    Value: 2 - plot all the clusters on seperate plots (subplot)
                                    Value: 3 - plot all the clusters on a single plot
        :param clusterName:        Name of the cluster to plot (if param:plotType is 1)
                                    Default: None
        :param additionalCriteria: List of desired criteria the plotted galaxies should meet
                                    Default: None
        :param useMembers:        Flag to indicate whether only cluster members should be plotted or only non-members should be plotted.
                                    Default: 'only' - only members
                                    Value:   'not' - only non-members
                                    Value:   'all' - no restriction imposed
        :param colorType:          Specifies how to color code the plotted galaxies
                                    Default: None
                                    Value:   'membership' - spectroscopic member vs photometric member
                                    Value:   'passive' - passive vs star forming
        :param colors:             Specifies what colors should be used when plotting
                                    Default: None - random colors are generated
                                    Value:   [(r,g,b), (r,g,b)]
        :param useStandards:       Flag to indicate whether the standard search criteria should be applied
                                    Default: True
        :param xRange:             List containing the desired lower and upper bounds for the x-axis
                                    Default: None
        :param yRange:             List containing the desired lower and upper bounds for the y-axis
                                    Default: None
        :param xLabel:             Label to put on the x-axis
                                    Default: Empty string
        :param yLabel:             Label to put on the y-axis
                                    Default: Empty string
        :param useLog:             Flag to indicate whether the x- or y-axis should be in log scale
                                    Default: [False,False] - neither axis in log scale
                                    Value:   [False,True] - y axis in log scale
                                    Value:   [True,False] - x axis in log scale
                                    Value:   [True,True] - both axis in log scale
        :param fitLine:             Flag to indicate whether a best fit line should be fit to the data. By default this line will plot size vs mass. 
                                     (note: the default x and y will be in log, however specifically selected values will correspond to the useLog list)
                                     (note: not currently configured to work with plot type 2)
        :post:                      The generated plot(s) will be displayed
        :return:                   (x, y), representing the total number of x-values and y-values corresponding to plotted data points
        """
        # Initialize plot
        plt.figure(figsize=(8,6))
        # Create 'goodData' flag for future checks
        self.reduceDF(additionalCriteria, useStandards)
        # Check if plot colors were provided by the user
        if colors != None:
            color1 = colors[0]
            color2 = colors[1]
        # If not, generate random colors
        else:
            if colorType == 'passive':
                color1 = "red"
                color2 = "blue"
            else:
                color1 = "green"
                color2 = "orange"
        # Plot only the cluster specified
        if plotType == 1:
            if clusterName == None:
                print("No cluster name provided!")
                return
            if useMembers == None:
                print("Please specify membership requirements!")
                return
            elif useMembers == 'all':
                # Get all galaxies associated with this cluster
                data = self.getClusterGalaxies(clusterName)
            elif useMembers == 'only':
                # Reduce data to only contain galaxies classified as members
                data = self.getMembers(clusterName)
            elif useMembers == 'not':
                # Reduce data to only contain galaxies not classified as members
                data = self.getNonMembers(clusterName)
            else:
                print(useMembers, " is not a valid membership requirement!")
                return
            # Plot data
            xTot, yTot = self.plotUnwrapped(xQuantityName, yQuantityName, colorType, useLog, fitLine, data, color1, color2, plt, bootstrap=bootstrap, plotErrBars=plotErrBars)
        # Plot all clusters individually in a subplot
        elif plotType == 2:
            # Initialize data count totals (used when running test suite)
            xTot = 0
            yTot = 0
            # Generate the subplots
            _, axes = plt.subplots(4,3,figsize=(15,12))
            currentIndex = 0
            # Loop over each subplot
            for i in range(4):
                for j in range(3):
                    # Exclude the 12th subplot (there are only 11 clusters in self.catalog)
                    if (currentIndex == len(self._structClusterNames)):
                        break
                    currentClusterName = self._structClusterNames[currentIndex]
                    if useMembers == None:
                        print("Please specify membership requirements!")
                        return
                    elif useMembers == 'all':
                        # Get all galaxies associated with this cluster
                        data = self.getClusterGalaxies(currentClusterName)
                    elif useMembers == 'only':
                        # Reduce data to only contain galaxies classified as members
                        data = self.getMembers(currentClusterName)
                    elif useMembers == 'not':
                        # Reduce data to only contain galaxies not classified as members
                        data = self.getNonMembers(currentClusterName)
                    else:
                        print(useMembers, " is not a valid membership requirement!")
                        return
                    # Plot data
                    x, y = self.plotUnwrapped(xQuantityName, yQuantityName, colorType, useLog, fitLine, data, color1, color2, axes[i][j], axes, i, j, bootstrap=bootstrap, plotErrBars=plotErrBars)
                    # Update data count totals
                    xTot+=x
                    yTot+=y
                    # Plot configurations for plotType 2
                    axes[i][j].set(xlabel=xLabel, ylabel=yLabel)
                    if (xRange != None):
                        axes[i][j].set(xlim=xRange)
                    if (yRange != None):
                        axes[i][j].set(ylim=yRange)
                    axes[i][j].set(title=currentClusterName)
                    if colorType != None:
                        # Avoid calling legend() if there are no labels
                        axes[i][j].legend()
                    currentIndex += 1
            # Remove the 12th subplot from the figure otherwise blank axes will be displayed
            plt.delaxes(axes[3][2])
            # Configure the subplot spacing so axes aren't overlapping
            # These specifc values were found at:
            # https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        # Plot all clusters on the same plot            
        elif plotType == 3:
            # Set an initial value to append to.
            data = pd.DataFrame()
            for clusterName  in self._structClusterNames:
                if useMembers == None:
                    print("Please specify membership requirements!")
                    return
                elif useMembers == 'all':
                    # Get all galaxies associated with this cluster
                    data = data.append(self.getClusterGalaxies(clusterName))
                elif useMembers == 'only':
                    # Reduce data to only contain galaxies classified as members
                    data = data.append(self.getMembers(clusterName))
                elif useMembers == 'not':
                    # Reduce data to only contain galaxies not classified as members
                    data = data.append(self.getNonMembers(clusterName))
                else:
                    print(useMembers, " is not a valid membership requirement!")
                    return
            xTot, yTot = self.plotUnwrapped(xQuantityName, yQuantityName, colorType, useLog, fitLine, data, color1, color2, plt, axes=None, row=None, col=None, bootstrap=bootstrap, plotErrBars=plotErrBars)
        else:
            print(plotType, " is not a valid plotting scheme!")
            return
        # Plot configurations for plotType 1 and 3
        # (plotType 2 handles plot configurations for each individual subplot)
        if plotType != 2:
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            if (xRange != None):
                plt.xlim(xRange[0], xRange[1])
            if (yRange != None):
                plt.ylim(yRange[0], yRange[1])
            if colorType != None:
                # Avoid calling legend() if there are no labels
                plt.legend()
            plt.show()
        return (xTot, yTot)
    # END PLOT