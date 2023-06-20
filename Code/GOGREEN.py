from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.markers as mrk
import numpy as np
import pandas as pd
import os
import warnings
import scipy.optimize as opt



class GOGREEN:
    def __init__(self, dataPath:str, priorCatalog:pd.DataFrame=pd.DataFrame(), usePhotMembership:bool=True):
        """
        __init__ Constructor to define and initialize class members

        :param dataPath:       absolute path to the directory containing DR1/ and STRUCTURAL_PARA_v1.1_CATONLY/
                                subdirectories
        :param priorCatalog:   If we already have our catalog up to date and recompiling to update something else, we pass it in as this parameter
                                Default: Empty DataFrame
        :param usePhotMembership: flag indicating whether photometric members should be included in the membership definition
                                Default: True
        """ 
        
        self.catalog = pd.DataFrame()
        self.sourceCatalog = pd.DataFrame()
        self.standardCriteria = []
        # Private Members
        self._path = dataPath
        self._structClusterNames = ['SpARCS0219', 'SpARCS0035','SpARCS1634', 'SpARCS1616', 'SPT0546', 'SpARCS1638',
                                    'SPT0205', 'SPT2106', 'SpARCS1051', 'SpARCS0335', 'SpARCS1034']
        self._clustersCatalog = pd.DataFrame()
        self._combinedSourceCatalog = pd.DataFrame()
        self._redshiftCatalog = pd.DataFrame()
        self._galfitCatalog = pd.DataFrame()
        self._matchedCatalog = pd.DataFrame()
        self._photSourceCatalog = pd.DataFrame()
        self._specSourceCatalog = pd.DataFrame()

        self.init(priorCatalog, usePhotMembership)
    # END __INIT__

    def init(self, priorCatalog:pd.DataFrame=None, usePhotMembership:bool=True):
        """
        init Helper method for initializing catalogs

        :param priorCatalog:   If we already have our catalog up to date and recompiling to update something else, it is passed in as this parameter
                                Default: None
        :param usePhotMembership: flag indicating whether photometric members should be included in the membership definition
                                Default: True
        """ 
        # Build path string to the cluster catalog
        clusterCatPath = self._path + 'DR1/CATS/Clusters.fits'
        # Generate a DataFrame of the catalog data
        self._clustersCatalog = self.generateDF(clusterCatPath)
        # Remove whitespaces included with some cluster names
        self._clustersCatalog['cluster'] = self._clustersCatalog['cluster'].str.strip()

        # Skip remaining set up if catalog already is up to date
        if not priorCatalog.empty:
            self.catalog = priorCatalog
            return

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

        # Merge source catalogs into a combined catalog
        merge_col = ['SPECID']
        # This only ouputs columns with names different than those in the photometric source table.  
        # Make sure that SPECID is added back in as we will match on that field
        cols_to_use = self._specSourceCatalog.columns.difference(self._photSourceCatalog.columns).tolist() + merge_col
        self._combinedSourceCatalog = self.merge(self._photSourceCatalog, self._specSourceCatalog[cols_to_use], merge_col)

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
            tempCPHOTID = self._combinedSourceCatalog[self._combinedSourceCatalog['Cluster'] == clusterName].iloc[0]['cPHOTID']
            # Extract the source ID and cluster ID from the temporary cPHOTID
            idPrefix = int(str(tempCPHOTID)[:3])*int(1e6)
            # Convert the structural catalog PHOTCATID into the photometric catalog cPHOTID
            matchedClusterDF.rename(columns = {'PHOTCATID':'cPHOTID'}, inplace = True)
            matchedClusterDF.loc[:,'cPHOTID'] += idPrefix
            # Combine it with the main struct matched DataFrame
            self._matchedCatalog = self._matchedCatalog.append(matchedClusterDF)

        # Merge combined source catalog with photomatched structural catalog by cPHOTID.
        self.catalog = self.merge(self._combinedSourceCatalog, self._matchedCatalog, 'cPHOTID')
        # readjust to preserve NaN values after merge
        self.catalog = self.catalog.replace(100000000000000000000, np.nan)
        # Generate cluster-centric distance columns
        self.calcClusterCentricDist()
        # Generate flags for use in plotting
        self.generateFlags()
        # Establish membership
        self.setMembers(usePhotMembership)
        self.setNonMembers(usePhotMembership)
        # Set error values (necessary because the re_err values from Galfit are not adequate)
        self.setReErr()
        # Generate converted unit columns (kpc instead of arcsec) for re values.
        self.reConvert()
        # Generate fractional error columns
        self.catalog['re_frac_err'] = self.catalog['re_err_robust']/self.catalog['re']
        self.catalog['re_frac_err_converted'] = self.catalog['re_err_robust_converted']/self.catalog['re_converted']
    # END INIT

    def generateFlags(self):
        """
        init Helper method for initializing flags relevant to plotting
        """ 
        # define queries
        #passiveQuery = '(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)' #(from van der Burg et al. 2020)
        passiveQuery = 'NUVMINV > 2 * VMINJ + 1.6' #(𝑁𝑈𝑉 − 𝑉 ) > 2(𝑉 − 𝐽) + 1.6(from McNab et al 2021)
        #starFormingQuery = '(UMINV <= 1.3) or (VMINJ >= 1.6) or (UMINV <= 0.60+VMINJ)' #(from van der Burg et al. 2020)
        starFormingQuery = 'NUVMINV < 2 * VMINJ + 1.1' #(𝑁𝑈𝑉 − 𝑉 ) < 2(𝑉 − 𝐽) + 1.1 (from McNab et al 2021)
        gvQuery = '(2 * VMINJ + 1.1 <= NUVMINV) and (NUVMINV <= 2 * VMINJ + 1.6)' # 2(𝑉 − 𝐽) + 1.1 ≤ (𝑁𝑈𝑉 − 𝑉 ) ≤ 2(𝑉 − 𝐽) + 1.6 (from McNab et al 2021)
        bqQuery = '((VMINJ + 0.45 <= UMINV) and (UMINV <= VMINJ + 1.35)) and ((-1.25 * VMINJ + 2.025 <= UMINV) and (UMINV <= -1.25 * VMINJ + 2.7))' # (𝑉 − 𝐽) + 0.45 ≤ (𝑈 − 𝑉 ) ≤ (𝑉 − 𝐽) + 1.35 ### − 1.25 (𝑉 − 𝐽) + 2.025 ≤ (𝑈 − 𝑉 ) ≤ −1.25 (𝑉 − 𝐽) + 2.7 (from McNab et al 2021)
        psbQuery = 'D4000 < 1.45 and delta_BIC < -10' # (D4000 < 1.45) ∩ (ΔBIC < −10) (from McNab et al 2021)
        ellipticalQuery = '2.5 < n < 6'
        spiralQuery = 'n <= 2.5'
        # Initialize flags to be used in plotting
        # Quality flag
        self.catalog['goodData'] = 1
        # Population flags
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

    def setMembers(self, usePhot:bool=True) -> pd.DataFrame:
        """
        setMembers Sets membership flag in the catalog based on criteria in McNab et. al. 2021

        :param usePhot: Flag indicating whether photometric members should be included in the membership definition
                                Default: True
        :return:            catalog is updated
        """
        # Intialize column
        self.catalog['member_adjusted'] = 0
        # McNab+21 criteria: (zq_spec>=3) & (member==1) ) | ( (( zq_spec<3) | (SPECID<0)) & (abs(zphot - zclust)<0.16)
        specZthreshold = (self.catalog['Redshift_Quality'] >= 3) & (self.catalog['member'] == 1) & (self.catalog['cluster_centric_distance_spec'] < 1000)
        if usePhot:
            photZthreshold = (np.abs(self.catalog['zphot'].values - self.catalog['Redshift'].values) < 0.16) & (self.catalog['cluster_centric_distance_phot'] < 1000)
            specZunderThreshold = (self.catalog['Redshift_Quality'] < 3) | (self.catalog['SPECID'] < 0)
            # Establish reduced dataset of members
            reduced = self.catalog[specZthreshold | ( specZunderThreshold & photZthreshold )]
        else: 
            # Exclude phot in membership definition if user elects to do this
            reduced = self.catalog[specZthreshold]
        # Update catalog
        self.catalog['member_adjusted'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)
    # END SETMEMBERS

    def setNonMembers(self, usePhot:bool=True) -> pd.DataFrame:
        """
        setNonMembers Sets non-membership flag in the catalog based on criteria in McNab et. al. 2021

        :param usePhot: Flag indicating whether photometric members should be included in the membership definition
                                Default: True
        :return:            catalog is updated
        """
        # Intialize column
        self.catalog['nonmember_adjusted'] = 0
        # McNab+21 criteria: ((zq_spec>=3) & (member==0) ) | ( (( zq_spec<3) | (SPECID<0)) & (abs(zphot - zclust)>=0.16))
        specZthreshold = (self.catalog['Redshift_Quality'] >= 3) & ((self.catalog['member'] == 0) | (self.catalog['cluster_centric_distance_spec'] >= 1000))
        if usePhot:
            photZthreshold = (np.abs(self.catalog['zphot'].values - self.catalog['Redshift'].values) >= 0.16) | (self.catalog['cluster_centric_distance_phot'] >= 1000)
            specZunderThreshold = (self.catalog['Redshift_Quality'] < 3) | (self.catalog['SPECID'] < 0)
            # Establish reduced dataset of members
            reduced = self.catalog[specZthreshold | ( specZunderThreshold & photZthreshold )]
        else:
            reduced = self.catalog[specZthreshold]
        # Update catalog
        self.catalog['nonmember_adjusted'] = self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)
    # END SETNONMEMBERS

    def setGoodData(self, additionalCriteria:list, useStandards:bool) -> pd.DataFrame:
        """
        setGoodData Reduces the catalog to contain only galaxies that meet the criteria provided in
                 param:additionalCriteria and the standard criteria (if param:useStandards is True)

        :param additionalCriteria: List of any criteria outside of standard to apply
        :param useStandards:       Flag to specify whether the standard criteria should be applied
        :return:                   Catalog is updated
        """
        # Reinitialize quality flag in case standards have changed since last calling, or additional criteria are provided
        self.catalog['goodData'] = 1
        # Set quality flags to false as indicated by additional criteria
        if (additionalCriteria != None):
            for criteria in additionalCriteria:
                reduced = self.catalog.query(criteria)
                self.catalog['goodData'] = self.catalog['goodData'] & (self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int))
        # Set quality flags to false as indicated by standard criteria
        if useStandards:
            for criteria in self.standardCriteria:
                reduced = self.catalog.query(criteria)
                self.catalog['goodData'] = self.catalog['goodData'] & (self.catalog['cPHOTID'].isin(reduced['cPHOTID']).astype(int)) # https://www.geeksforgeeks.org/python-pandas-series-astype-to-convert-data-type-of-series/
    # END SETGOODDATA
        
    def getClusterGalaxies(self, clusterName:str) -> pd.DataFrame:
        """
        getClusterGalaxies Get all galaxies associated with the cluster provided by param:clusterName

        :param clusterName: Name of the cluster whose galaxies should be returned
        :return:            Pandas DataFrame containing only galaxies associated with cluster param:clusterName 
        """
        return self.catalog[self.catalog['Cluster'] == clusterName]
    # END GETCLUSTERGALAXIES

    def plotPassiveLines(self, axes:list=None, row:int=None, col:int=None):
        """
        plotPassiveLines Draws the recognized boundary between passive and star-forming galaxies on UVJ plots

        :param axes:                The array of subplots created when the plotType is set to 2.
                                     Default: None
        :param row :                Specifies the row of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :param col :                Specifies the column of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :return    :                lines are plotted
        """
        # Generate the data points used to plot the line
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
        plotVanDerWelLines plots the passive and star-forming MSR trends calculated in van der Wel et al. 2014

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
        xVals = np.array([9.8, 11.5]) # mass values in log space
        MstellarRange = pow(10, xVals) # mass values in linear space
        # Calculate size values (output in log space)
        yValsPassive = np.log10(np.array([Apassive * pow((i / (5 * np.float_power(10, 10))), alphaPassive) for i in MstellarRange]))
        yValsSF = np.log10(np.array([Asf * pow((i / (5 * np.float_power(10, 10))), alphaSF) for i in MstellarRange]))
        # Plot lines in log space
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
        plotMcNabPlots plots three plots and a table from McNab et. al. 2021

        :return    :     plots are plotted
        """
        # Establish appropriate standard criteria
        searchCriteria = [
            'Star == 0',
            'K_flag < 4',
            'Mstellar > 10**9.5',
            '(1 < zspec < 1.5) or (((Redshift_Quality < 3) or (SPECID < 0)) and (1 < zphot < 1.5))',
            'cluster_id <= 12'
        ]
        self.standardCriteria = searchCriteria
        # Set data quality flags according to standard criteria
        self.setGoodData(None, True)
        print("Total phot sample: " + str(self.catalog.query('cluster_id <= 12 and zphot > 1 and zphot < 1.5 and K_flag >= 0 and K_flag < 4 and Star == 0 and Mstellar > 10**9.5').shape[0]) + " - expected: 3062")
        print("Total spec sample: " + str(self.catalog.query('cluster_id <= 12 and zspec > 1 and zspec < 1.5 and Redshift_Quality >= 3 and Star == 0').shape[0]) + " - expected: 722")
        print("Total spec sample (above mass limit): " + str(self.catalog.query('cluster_id <= 12 and cluster_id >= 1 and zspec > 1 and zspec < 1.5 and Redshift_Quality >= 3 and Star == 0 and Mstellar > 10**10.2').shape[0]) + " - expected: 342")
        # Construct table
        table = pd.DataFrame()
        table['Population'] = ['SF', 'Q', 'GV', 'BQ', 'PSB']
        table['Total Sample'] = [self.catalog.query('(member_adjusted == 1 or nonmember_adjusted == 1) and goodData == 1 and starForming == 1').shape[0], 
            self.catalog.query('(member_adjusted == 1 or nonmember_adjusted == 1) and goodData == 1 and passive == 1').shape[0], 
            self.catalog.query('(member_adjusted == 1 or nonmember_adjusted == 1) and goodData == 1 and greenValley == 1').shape[0], 
            self.catalog.query('(member_adjusted == 1 or nonmember_adjusted == 1) and goodData == 1 and blueQuiescent == 1').shape[0], 
            self.catalog.query('(member_adjusted == 1 or nonmember_adjusted == 1) and goodData == 1 and postStarBurst == 1').shape[0]]
        table['Cluster Members'] = [self.catalog.query('member_adjusted == 1 and goodData == 1 and starForming == 1').shape[0], 
            self.catalog.query('member_adjusted == 1 and goodData == 1 and passive == 1').shape[0], 
            self.catalog.query('member_adjusted == 1 and goodData == 1 and greenValley == 1').shape[0], 
            self.catalog.query('member_adjusted == 1 and goodData == 1 and blueQuiescent == 1').shape[0], 
            self.catalog.query('member_adjusted == 1 and goodData == 1 and postStarBurst == 1').shape[0]]
        expectedTable = pd.DataFrame()
        expectedTable['Population'] = ['SF', 'Q', 'GV', 'BQ', 'PSB']
        expectedTable['Total Sample'] = [1302, 702, 257, 164, 54]
        expectedTable['Cluster Members'] = [463, 504, 125, 106, 34]
        diffTable = pd.DataFrame()
        diffTable['Population'] = ['SF', 'Q', 'GV', 'BQ', 'PSB']
        diffTable['Total Sample'] = table['Total Sample'] - expectedTable['Total Sample']
        diffTable['Cluster Members'] = table['Cluster Members'] - expectedTable['Cluster Members']

        # Display table
        print("Table 2")
        print(table)
        print("Table 2 - Expected")
        print(expectedTable)
        print("Table 3 - Difference")
        print(diffTable)
        # Extract desired quantities from data for plot
        # "Bad" populations refer to those before the spectroscopic mass threshold (to be displayed with smaller points)
        passiveMembersBad = self.catalog.query('member_adjusted == 1 and goodData == 1 and passive == 1 and Mstellar <= 10**10.2')
        starFormingMembersBad = self.catalog.query('member_adjusted == 1 and goodData == 1 and starForming == 1 and Mstellar <= 10**10.2')
        greenValleyMembersBad = self.catalog.query('member_adjusted == 1 and goodData == 1 and greenValley == 1 and Mstellar <= 10**10.2')
        blueQuiescentMembersBad = self.catalog.query('member_adjusted == 1 and goodData == 1 and blueQuiescent == 1 and Mstellar <= 10**10.2')
        postStarBurstMembersBad = self.catalog.query('member_adjusted == 1 and goodData == 1 and postStarBurst == 1 and Mstellar <= 10**10.2')

        passiveMembersGood = self.catalog.query('member_adjusted == 1 and goodData == 1 and passive == 1 and Mstellar > 10**10.2')
        starFormingMembersGood = self.catalog.query('member_adjusted == 1 and goodData == 1 and starForming == 1 and Mstellar > 10**10.2')
        greenValleyMembersGood = self.catalog.query('member_adjusted == 1 and goodData == 1 and greenValley == 1 and Mstellar > 10**10.2')
        blueQuiescentMembersGood = self.catalog.query('member_adjusted == 1 and goodData == 1 and blueQuiescent == 1 and Mstellar > 10**10.2')
        postStarBurstMembersGood = self.catalog.query('member_adjusted == 1 and goodData == 1 and postStarBurst == 1 and Mstellar > 10**10.2')

        # Construct plot 1
        plt.figure()
        # Plot "bad" data
        plt.scatter(passiveMembersBad['VMINJ'], passiveMembersBad['NUVMINV'], alpha=0.5, s=8, marker='o', color='red')
        plt.scatter(starFormingMembersBad['VMINJ'], starFormingMembersBad['NUVMINV'], alpha=0.5, s=8, marker='*',  color='blue')
        plt.scatter(greenValleyMembersBad['VMINJ'], greenValleyMembersBad['NUVMINV'], alpha=0.5, s=8, marker='d', color='green')
        plt.scatter(blueQuiescentMembersBad['VMINJ'], blueQuiescentMembersBad['NUVMINV'], alpha=0.5, s=30, marker='s', color='orange')
        plt.scatter(postStarBurstMembersBad['VMINJ'], postStarBurstMembersBad['NUVMINV'], alpha=0.5, s=30, marker='x', color='purple')
        # Plot "good" data
        plt.scatter(passiveMembersGood['VMINJ'], passiveMembersGood['NUVMINV'], alpha=0.5, s=30, marker='o', color='red')
        plt.scatter(starFormingMembersGood['VMINJ'], starFormingMembersGood['NUVMINV'], alpha=0.5, s=30, marker='*',  color='blue')
        plt.scatter(greenValleyMembersGood['VMINJ'], greenValleyMembersGood['NUVMINV'], alpha=0.5, s=30, marker='d', color='green')
        plt.scatter(blueQuiescentMembersGood['VMINJ'], blueQuiescentMembersGood['NUVMINV'], alpha=0.5, s=60, marker='s', color='orange', label='BQ')
        plt.scatter(postStarBurstMembersGood['VMINJ'], postStarBurstMembersGood['NUVMINV'], alpha=0.5, s=60, marker='x', color='purple', label='PSB')
        # Indicate the green valley region
        plt.plot([0.2, 2], [2, 5.5], linestyle='dashed', color='black')
        plt.plot([0.2, 2], [1.5, 5], linestyle='dashed', color='black')
        plt.fill_between([0.2, 2], [1.5, 5], [2, 5.5], color='green', alpha=0.1)
        # Format plot 1
        plt.xlabel("(V-J)")
        plt.ylabel("(NUV-V)")
        plt.xlim(0.2, 2)
        plt.ylim(1, 6)
        plt.title("Figure 1a")
        plt.legend()

        # Construct plot 2
        plt.figure()
        # Plot "bad" data
        plt.scatter(passiveMembersBad['VMINJ'], passiveMembersBad['UMINV'], alpha=0.5, s=8, marker='o', color='red')
        plt.scatter(starFormingMembersBad['VMINJ'], starFormingMembersBad['UMINV'], alpha=0.5, s=8, marker='*',  color='blue')
        plt.scatter(greenValleyMembersBad['VMINJ'], greenValleyMembersBad['UMINV'], alpha=0.5, s=30, marker='d', color='green')
        plt.scatter(blueQuiescentMembersBad['VMINJ'], blueQuiescentMembersBad['UMINV'], alpha=0.5, s=8, marker='s', color='orange')
        plt.scatter(postStarBurstMembersBad['VMINJ'], postStarBurstMembersBad['UMINV'], alpha=0.5, s=30, marker='x', color='purple')
        # Plot "good" data
        plt.scatter(passiveMembersGood['VMINJ'], passiveMembersGood['UMINV'], alpha=0.5, s=30, marker='o', color='red', label='Q')
        plt.scatter(starFormingMembersGood['VMINJ'], starFormingMembersGood['UMINV'], alpha=0.5, s=30, marker='*',  color='blue', label='SF')
        plt.scatter(greenValleyMembersGood['VMINJ'], greenValleyMembersGood['UMINV'], alpha=0.5, s=60, marker='d', color='green', label='GV')
        plt.scatter(blueQuiescentMembersGood['VMINJ'], blueQuiescentMembersGood['UMINV'], alpha=0.5, s=30, marker='s', color='orange')
        plt.scatter(postStarBurstMembersGood['VMINJ'], postStarBurstMembersGood['UMINV'], alpha=0.5, s=60, marker='x', color='purple')
        #xPoints = [0.3, 0.6, 0.7, 1]
        #yPoints = [1.65, 1.95, 1.2, 1.45]
        # Indicate the blue quiescent region
        plt.plot([0.3, 0.6], [1.65, 1.95], linestyle='dashed', color='black') # top left
        plt.plot([0.7, 1], [1.2, 1.45], linestyle='dashed', color='black') # bottom right
        plt.plot([0.6, 1], [1.95, 1.45], linestyle='dashed', color='black') # top right
        plt.plot([0.3, 0.7], [1.65, 1.2], linestyle='dashed', color='black') # bottom left
        plt.fill_between([0.3, 0.6], [1.65, 1.3], [1.65, 1.95], color='orange', alpha=0.1)
        plt.fill_between([0.7, 1], [1.2, 1.45], [1.85, 1.45], color='orange', alpha=0.1)
        plt.fill_between([0.6, 0.7], [1.3, 1.2], [1.95, 1.85], color='orange', alpha=0.1)
        # Format plot 2
        plt.xlabel("(V-J)")
        plt.ylabel("(U-V)")
        plt.xlim(0.25, 2.25)
        plt.ylim(0.5, 2.6)
        plt.title("Figure 1b")
        plt.legend()

        # Construct plot 3
        plt.figure()
        # Plot "bad" data
        plt.scatter(passiveMembersBad['D4000'], passiveMembersBad['UMINV'], alpha=0.5, s=8, marker='o', color='red')
        plt.scatter(starFormingMembersBad['D4000'], starFormingMembersBad['UMINV'], alpha=0.5, s=8, marker='*',  color='blue')
        plt.scatter(greenValleyMembersBad['D4000'], greenValleyMembersBad['UMINV'], alpha=0.5, s=8, marker='d', edgecolor='black', color='green')
        plt.scatter(blueQuiescentMembersBad['D4000'], blueQuiescentMembersBad['UMINV'], alpha=0.5, s=30, marker='s', color='orange')
        plt.scatter(postStarBurstMembersBad['D4000'], postStarBurstMembersBad['UMINV'], alpha=0.5, s=30, marker='x', edgecolor='black', color='purple')
        # Plot "good" data
        plt.scatter(passiveMembersGood['D4000'], passiveMembersGood['UMINV'], alpha=0.5, s=30, marker='o', color='red', label='Q')
        plt.scatter(starFormingMembersGood['D4000'], starFormingMembersGood['UMINV'], alpha=0.5, s=30, marker='*',  color='blue', label='SF')
        plt.scatter(greenValleyMembersGood['D4000'], greenValleyMembersGood['UMINV'], alpha=0.5, s=30, marker='d', edgecolor='black', color='green', label='GV')
        plt.scatter(blueQuiescentMembersGood['D4000'], blueQuiescentMembersGood['UMINV'], alpha=0.5, s=60, marker='s', color='orange', label='BQ')
        plt.scatter(postStarBurstMembersGood['D4000'], postStarBurstMembersGood['UMINV'], alpha=0.5, s=60, marker='x', edgecolor='black', color='purple', label='PSB')
        # Format plot 3
        plt.xlabel("D4000")
        plt.ylabel("(U-V)")
        plt.xlim(0.9, 2.2)
        plt.ylim(0.5, 2.1)
        plt.title("Figure 3")
        plt.legend()
    #END PLOTMCNABPLOTS

    def setReErr(self):
        """
        Assign error values for Effective Radius (Re) measurements. These values are taken from van der Wel et. al. 2012, Table 3

        :return   :    Catalog is updated
        """
        # Initialize to NaN
        self.catalog['re_err_robust'] = np.nan
        # Designate discrete error values corresponding to magnitude and effective radius (in arcseconds)
        # We take a conservative estimate and assume all galaxies with magnitude brighter than 21 have the same error as those with a magnitude of 21
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) <= 21) & (self.catalog.re < 0.3), self.catalog.re*0.01, self.catalog.re_err_robust) #https://stackoverflow.com/questions/12307099/modifying-a-subset-of-rows-in-a-pandas-dataframe
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) <= 21) & (self.catalog.re > 0.3), self.catalog.re*0.01, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 22) & (self.catalog.re < 0.3), self.catalog.re*0.02, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 22) & (self.catalog.re > 0.3), self.catalog.re*0.02, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 23) & (self.catalog.re < 0.3), self.catalog.re*0.03, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 23) & (self.catalog.re > 0.3), self.catalog.re*0.06, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 24) & (self.catalog.re < 0.3), self.catalog.re*0.08, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 24) & (self.catalog.re > 0.3), self.catalog.re*0.15, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 25) & (self.catalog.re < 0.3), self.catalog.re*0.18, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 25) & (self.catalog.re > 0.3), self.catalog.re*0.33, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 26) & (self.catalog.re < 0.3), self.catalog.re*0.42, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 26) & (self.catalog.re > 0.3), self.catalog.re*0.63, self.catalog.re_err_robust)
        self.catalog['re_err_robust'] = np.where((np.round(self.catalog.mag) == 27) & (self.catalog.re < 0.3), self.catalog.re*0.76, self.catalog.re_err_robust)
    #END SETREERR

    def calcClusterCentricDist(self):
        """
        Assign cluster-centric distance values for all member galaxies

        :return   :    Catalog is updated
        """
        # Initialize to NaN
        self.catalog['cluster_z'] = np.nan
        self.catalog['cluster_centric_distance_phot'] = np.nan
        self.catalog['cluster_centric_distance_spec'] = np.nan
        # Assign cluster RA and DEC values
        structClusters = ['SpARCS0219', 'SpARCS0035','SpARCS1634', 'SpARCS1616', 'SPT0546', 'SpARCS1638',
                                    'SPT0205', 'SPT2106', 'SpARCS1051', 'SpARCS0335', 'SpARCS1034']
        cluster_Redshifts = []
        # Add in cluster redshifts
        for clusterName in structClusters:
            cluster_Redshifts.append(float(self._clustersCatalog[self._clustersCatalog['cluster'] == clusterName].Redshift))
        # Fill in cluster RA and DEC columns in catalog for ease of access
        for i in range(0, len(structClusters)):
            self.catalog['cluster_z'] = np.where(self.catalog.cluster == structClusters[i], cluster_Redshifts[i], self.catalog.cluster_z)
        # Calculate cluster-centric distance for each member and fill in column
        self.catalog['cluster_centric_distance_phot'] = self.ccd(self.catalog.ra, self.catalog.dec, self.catalog.RA_Best, self.catalog.DEC_Best, self.catalog.cluster_z)
        self.catalog['cluster_centric_distance_spec'] = self.ccd(self.catalog['RA(J2000)'].values, self.catalog['DEC(J2000)'], self.catalog.RA_Best, self.catalog.DEC_Best, self.catalog.cluster_z)

        """
        ccd Helper function called by calcClusterCentricDist()

        :param gal_RA   :      Right ascension of each galaxy (in degrees)
        :param clust_RA :      Right ascension of each galaxy's cluster (in degrees)
        :param gal_DEC  :      Declination of each galaxy (in degrees)
        :param clust_DEC:      Declination of each galaxy's cluster (in degrees)
        :param clust_z  :      Redshift of each galaxy's cluster (z)
        :param is_member:      Flag indicating whether the galaxy is a member of the cluster associated with it.
                                Value: 1 - indicates true
                                Value: 0 - indicates false
        :return         :      Cluster-centric distance of each galaxy (in kpc)
        """
    def ccd(self, gal_RA, gal_DEC, clust_RA, clust_DEC, clust_z):
        # Convert cluster dec to radians
        clust_DEC_rad = clust_DEC * np.pi / 180
        # Calculate cluster-centric distance in degrees
        ccd_deg = np.sqrt(pow((gal_RA - clust_RA)*np.cos(clust_DEC_rad), 2) + pow(gal_DEC - clust_DEC, 2))
        # Convert to arcmin
        ccd_arcmin = ccd_deg*60
        # Convert to kpc
        ccd_arcmin_vals = ccd_arcmin.values
        clust_z_vals = clust_z.values
        ccd_kpc_vals = []
        for i in range(0, len(ccd_arcmin_vals)):
            if clust_z_vals[i] >= 0: # We avoid inputting NaN values to the conversion function
                ccd_kpc_vals.append(ccd_arcmin_vals[i]*cosmo.kpc_proper_per_arcmin(clust_z_vals[i])) # Have to input single values to the conversion function
                # Remove units
                ccd_kpc_vals[i] = (ccd_kpc_vals[i] / u.kpc) * u.arcmin
            else:
                ccd_kpc_vals.append(np.NaN)
        # Unsquare
        return ccd_kpc_vals
                                            

    def reConvert(self):
        """
        reConvert Create new columns in the catalog for effective radius and its error in units of kpc.

        :return   :    Catalog is updated

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
        # Add columns
        self.catalog['re_converted'] = sizes_converted
        self.catalog['re_err_robust_converted'] = errs_converted
    #END RECONVERT

    def MSRfit(self, data:list, useLog:list=[False, False], axes:list=None, row:int=None, col:int=None, typeRestrict:str=None, color:str=None, bootstrap:bool=True) -> tuple:
        """
        MSRfit fits a best fit line to data generated by the plot() method

        :param data:                The set of catalog data that is relevant to the plot already generated by the plot() method
        :param useLog:              Flag to indicate whether the x- or y-axis should be in log scale (always determines whether fit is done in linear or log space)
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
        :param typeRestrict:        Name indicating what population the trend is for, for use in constructing the label
        :param color1:              The color the fit line should be.
                                     Default: None     
        :param bootstrap            Flag to indicate rather bootstrapping should be used to calculate and display uncertainty on the fit 
                                     Default: True
        :return   :

        """
        # Establish label
        if typeRestrict == None:
            lbl = "stellar mass-size relation trend"
        else:
            lbl = typeRestrict + " stellar mass-size relation trend"
        # Extract values from data (note that these will all be in linear space as this data is taken straight from the catalog)
        size = data['re_converted'].values
        mass = data['Mstellar'].values
        errs = data['re_err_robust_converted'].values
        # Convert to fractional error
        errs = errs/size
        # Calculate coefficients using line-fitting algorithm (output will be in log space)
        print(typeRestrict + " count: " + str(mass.shape[0]))
        s, _ = opt.curve_fit(f=lambda x, m, b: pow(10, m*np.log10(x) + b), xdata=mass, ydata=size, sigma=errs, bounds=([-10, -10], [10, 10]), loss="huber") 
        slope = s[0]
        intercept = s[1]
        uncertainty = None
        # Define x bounds (in log space)
        xBounds = np.array([9.8,11.5])
        # Plot lines
        if row != None and col != None:
            # Check for subplots
            if axes[row][col] != None:
                if bootstrap:
                    # Bootstrapping calculation
                    uncertainty = self.bootstrap(mass, size, errs, axes, row, col, lineColor=color)
                # Add white backline in case of plotting multiple fit lines in one plot
                if color != 'black':
                    axes[row][col].plot(xBounds, intercept + slope*xBounds, color='white', linewidth=4)
                # Plot the best fit line (in log space)
                axes[row][col].plot(xBounds, intercept + slope*xBounds, color=color, label=lbl)
                return
        if bootstrap:
            # Bootstrapping calculation
            uncertainty = self.bootstrap(mass, size, errs, axes, row, col, lineColor=color)
        # Add white backline in case of plotting multiple fit lines in one plot
        if color != 'black':
            plt.plot(xBounds, intercept + slope*xBounds, color='white', linewidth=4)
        # Plot the best fit line (in log space)
        plt.plot(xBounds, intercept + slope*xBounds, color=color, label=lbl)
        return (slope, intercept, uncertainty)
    # END MSRFIT

    def bootstrap(self, x:list=None, y:list=None, error:list=None, axes:list=None, row:int=None, col:int=None, lineColor:str=None) -> tuple:
        """
        bootstrap Obtains a measure of error of the line-fitting algorithm we use.
        
        :param x:                   List containing the mass values of the data set (in linear space)
                                     Default: None
        :param y:                   List containing the size values corresponding to each mass value in the data set (in linear space)
                                     Default: None
        :param error:               List containing the error values corresponding to each size value in the data set (in linear space)
                                     Default: None
        :param axes:                The array of subplots created when the plotType is set to 2.
                                     Default: None
        :param row:                 Specifies the row of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :param col:                 Specifies the column of the 2D array of subplots. For use when axes is not None.
                                     Default: None
        :param lineColor:           Flag to indicate what color should be used to accentuate the trendline.
                                     Default: None
        :return      :    bootstrap uncertainty region is plotted.
        """
        # Initialize type of plot
        plot = plt
        # Check for subplots
        if row != None and col != None:
            # Check for subplots
            if axes[row][col] != None:
                plot = axes[row][col]
        # Initialize seed for consistent results across runs
        rng = np.random.RandomState(1234567890) # reference: https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array
        # Store size of x data
        size = len(x)
        # Bounds of x data stored in log space
        xMin = np.log10(np.min(x))
        xMax = np.log10(np.max(x))
        # Initialize arrays
        slopes = np.empty((100,))
        intercepts = np.empty((100,))
        # Create 100 bootstrap lines
        for i in range(0, 100):
            plotted = False
            while not(plotted):
                # Initialize new array of synthetic data
                randIndices = rng.randint(0, size, size=size)
                # Fill mutatedX with randomly selected mass values from x (in linear space)
                bootstrapX = x[randIndices]
                bootstrapY = y[randIndices]
                boostrapE = error[randIndices]
                # Fit data with equation (in try-catch block to help detect errors)
                try:
                    # Calculate coefficients for a bootstrap line (output will be in log space)
                    s, _ = opt.curve_fit(f=lambda x, m, b: pow(10, m*np.log10(x) + b), xdata=bootstrapX, ydata=bootstrapY, sigma=boostrapE, bounds=([-10, -10], [10, 10]), loss="huber")
                    m = s[0]
                    b = s[1]
                    # Store coefficients
                    intercepts[i] = b
                    slopes[i] = m
                    # Uncomment to plot each bootstrap line (in log space):
                    #xline = np.array([xMin, xMax])
                    #yline = b + m*xline
                    #plot.plot(xline, yline, color='green', alpha=0.6)
                    plotted = True
                except RuntimeError:
                    print("caught runtime error")
                except np.linalg.LinAlgError:
                    print("caught linear algebra error")
        # Create grid of points to test calculated m & b values at (in log space).
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
        elif lineColor == 'yellow':
            color = [1, 1, 0.8] # lighter yellow
        elif lineColor == 'purple':
            color = [0.8, 0.6, 1] # lighter purple
        # star-forming and default case
        else:
            color = [0, 0, 0.5] # darker blue
        # Plot curves on top and bottom of intervals
        plot.plot(xGrid, yTops, color=color)
        plot.plot(xGrid, yBots, color=color)
        # Fill in region
        plot.fill_between(xGrid, yBots, yTops, color=color, alpha=0.5) # https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html
        # Return uncertainty boundaries
        return (xGrid, yGrid)
    # END BOOTSTRAP

    def evalLineFit(self):
        """
        evalLineFit Evaluates the accuracy of the line-fitting algorithm used in this class to determine the stellar mass-size relation.
        
        :return: estimated slope and intercept are displayed in comparison to the actual slope and intercept.
        """
        # log(y) = 1.213log(x) - 2.44  <- arbitrary line equation we choose to test
        m = 1.213
        b = -2.44
        xBounds = np.array([9.8,11.5])
        # Specify consistent random seed
        rng = np.random.RandomState(1234567890)
        # Create random fractional uncertainty
        randUncCap = rng.random()/2
        # Create set of 100 fake data points
        nFake = [[], [], []]
        for i in range(0, 100):
            # Create random x value (in log space) between 9.8 and 11.5
            randXLineLg = 0.0001*rng.randint(xBounds[0]*10000, xBounds[1]*10000) # A wack but simple way to generate numbers with appropriate accuracy in a given range
            # Convert x to linear space
            randXLine = pow(10, randXLineLg)
            # Create y value (in log space) corresponding to our x on the arbitrary line
            yLineLg = m*randXLineLg + b
            # Convert y to linear space
            yLine = pow(10, yLineLg)
            # Create fake data point using a normal distribution of uncertainties
            yFake = rng.normal(loc=yLine, scale=randUncCap)
            # Store our random x, random y, and uncertainty (non-fractional) on the random y
            nFake[0].append(randXLine)
            nFake[1].append(yFake)
            nFake[2].append(yFake*randUncCap)
        # Fit a line through our fake data points (output will be in log space)
        s, _ = opt.curve_fit(f=lambda x, m, b: pow(10, m*np.log10(x) + b), xdata=nFake[0], ydata=nFake[1], sigma=nFake[2], bounds=([-10, -10], [10, 10]), loss="huber")
        # Extract results
        slope = s[0]
        intercept = s[1]
        # Show fake data points (in log space)
        plt.scatter(np.log10(nFake[0]), np.log10(nFake[1]))
        # Show calculated trend line (in log space)

        plt.plot(xBounds, intercept + slope*xBounds)
        # Display results
        print("Actual values: slope = " + str(m) + ", intercept = " + str(b))
        print("Estimated values: slope = " + str(slope) + ", intercept = " + str(intercept))
        # We can't print the difference here because the values are so close that cancellation error dominates
    # END EVALLINEFIT
        

    def compTrends(self, x:float=None, y:float=None, bootstrap:bool=True, limitRange:bool=True, plotType:str="default") -> tuple:
        """
        compTrends Calculates the ratio of member over non-member galaxies and plots relevant trendlines.
        :param x:           X value at which the comparison should be made
                             Default: None
        :param y:           Y value at which the comparison should be made
                             Default: None
        :param bootstrap:   Flag to indicate rather bootstrapping should be used to calculate and display uncertainty on the fit 
                             Default: True
        :param plotType:    Determines specific behavior of the function
                             Value: "default" - quiescent and star-forming ratios are calculated separately
                             Value: "transition" - transition lines are plotted as well
                             Value: "lit" - whole population's ratio is calculated and compared to the literature.
        :return: tuple of up to two distinct population ratios of member over non-member galaxies, plus between 2 and 7 trendlines are plotted
        """
        # Adjust plot size
        plt.figure(figsize=(10,10))
        # Reduce according to standard criteria
        self.setGoodData(None, True)
        # Quiescent and star-forming trends for default or transition option
        if plotType == "default" or plotType == "transition":
            passiveMembers = self.catalog.query('member_adjusted == 1 and passive == 1 and goodData == 1')
            passiveNonMembers = self.catalog.query('nonmember_adjusted == 1 and passive == 1 and goodData == 1')
            starFormingMembers = self.catalog.query('member_adjusted == 1 and starForming == 1 and goodData == 1')
            starFormingNonMembers = self.catalog.query('nonmember_adjusted == 1 and starForming == 1 and goodData == 1')
            # Plot quiescent and sf trends for members and nonmembers (4 lines total)
            mMemberQ, bMemberQ, uncMemberQ = self.MSRfit(data=passiveMembers, useLog=[True, True], typeRestrict='Quiescent cluster', color='red', bootstrap=bootstrap)
            mMemberSF, bMemberSF, uncMemberSF = self.MSRfit(data=starFormingMembers, useLog=[True, True], typeRestrict='Star-Forming cluster', color='blue', bootstrap=bootstrap)
            mNonMemberQ, bNonMemberQ, uncNonMemberQ = self.MSRfit(data=passiveNonMembers, useLog=[True, True], typeRestrict='Quiescent field', color='orange', bootstrap=bootstrap)
            mNonMemberSF, bNonMemberSF, uncNonMemberSF = self.MSRfit(data=starFormingNonMembers, useLog=[True, True], typeRestrict='Star-Forming field', color='green', bootstrap=bootstrap)
            # Analyze bootstrap difference
            if plotType == "default":
                # Format plot
                plt.legend()
                # Handle bootstrap analysis
                if bootstrap:
                    # Extract grid of x values used for bootstrap.
                    xVals = uncMemberQ[0].tolist()
                    # Calculate diffs
                    diffsQ = self.getBootstrapDiffs(uncMemberQ, uncNonMemberQ)
                    diffsSF = self.getBootstrapDiffs(uncMemberSF, uncNonMemberSF)
                    # Plot mass vs diffs
                    plt.figure()
                    plt.plot(xVals, diffsSF[0], color='blue')
                    plt.figure()
                    plt.plot(xVals, diffsQ[0], color='red')
                    # Plot mass vs diff confidence intervals
                    plt.figure()
                    plt.plot(xVals, diffsSF[1], color='blue')
                    plt.plot(xVals, diffsSF[2], color='blue')
                    plt.figure()
                    plt.plot(xVals, diffsQ[1], color='red')
                    plt.plot(xVals, diffsQ[2], color='red')
            # Transition galaxy option
            elif plotType == "transition":
                # Extracted desired quantities from data
                gvMembers = self.catalog.query('member_adjusted == 1 and greenValley == 1 and goodData == 1')
                gvNonMembers = self.catalog.query('nonmember_adjusted == 1 and greenValley == 1 and goodData == 1')
                bqMembers = self.catalog.query('member_adjusted == 1 and blueQuiescent == 1 and goodData == 1')
                bqNonMembers = self.catalog.query('nonmember_adjusted == 1 and blueQuiescent == 1 and goodData == 1')
                psbMembers = self.catalog.query('member_adjusted == 1 and postStarBurst == 1 and goodData == 1')
                psbNonMembers = self.catalog.query('nonmember_adjusted == 1 and postStarBurst == 1 and goodData == 1')
                # Plot trends (6 additional lines)
                self.MSRfit(data=gvMembers, useLog=[True, True], typeRestrict='GV cluster', color='purple', bootstrap=bootstrap)
                self.MSRfit(data=gvNonMembers, useLog=[True, True], typeRestrict='GV field', color='pink', bootstrap=bootstrap)
                self.MSRfit(data=bqMembers, useLog=[True, True], typeRestrict='BQ cluster', color='black', bootstrap=bootstrap)
                self.MSRfit(data=bqNonMembers, useLog=[True, True], typeRestrict='BQ field', color='gray', bootstrap=bootstrap)
                self.MSRfit(data=psbMembers, useLog=[True, True], typeRestrict='PSB cluster', color='brown', bootstrap=bootstrap)
                self.MSRfit(data=psbNonMembers, useLog=[True, True], typeRestrict='PSB field', color='yellow', bootstrap=bootstrap)
                # Format plot
                plt.legend() 
            # Ratio calculation for default or transition option
            # if x or y values are provided, return ratio at that value
            if x != None and y == None:
                # Get ratios at a certain x value
                pointMemberQ = x*mMemberQ + bMemberQ
                pointMemberSF = x*mMemberSF + bMemberSF 
                pointNonMemberQ = x*mNonMemberQ + bNonMemberQ 
                pointNonMemberSF = x*mNonMemberSF + bNonMemberSF
                ratioQ = pointMemberQ/pointNonMemberQ
                ratioSF = pointMemberSF/pointNonMemberSF
            elif y != None and x == None:
                # Get ratios at a certain y value
                pointMemberQ = (y/mMemberQ) - (bMemberQ/mMemberQ)
                pointMemberSF = (y/mMemberSF) - (bMemberSF/mMemberSF)
                pointNonMemberQ = (y/mNonMemberQ) - (bNonMemberQ/mNonMemberQ) 
                pointNonMemberSF = (y/mNonMemberSF) - (bNonMemberSF/mNonMemberSF) 
                ratioQ = pointMemberQ/pointNonMemberQ
                ratioSF = pointMemberSF/pointNonMemberSF
            # Error cases
            elif x != None and y != None:
                print("Error: Both x and y values were provided. Please provide only one.")
                return (np.nan, np.nan)    
            else:
                print("Error: No point of comparison provided. Please provide an x or y value to test the ratio of.")
                return (np.nan, np.nan)  
            # Return a tuple containing the ratios
            return (ratioQ, ratioSF)
        # Ratio calculation for lit option
        if plotType == "lit":
            members = self.catalog.query('member_adjusted == 1 and goodData == 1')
            nonMembers = self.catalog.query('nonmember_adjusted == 1 and goodData == 1')
            mMember, bMember, _ = self.MSRfit(data=members, useLog=[True, True], typeRestrict='cluster', color="green", bootstrap=bootstrap)
            mNonMember, bNonMember, _ = self.MSRfit(data=nonMembers, useLog=[True, True], typeRestrict='field', color="orange", bootstrap=bootstrap)
            # if x or y values are provided, return ratio at that value
            if x != None and y == None:
                # Get ratios at a certain x value
                pointMember = x*mMember + bMember
                pointNonMember = x*mNonMember + bNonMember
                ratio = pointMember/pointNonMember
                diff = pointMember - pointNonMember
            elif y != None and x == None:
                # Get ratios at a certain y value
                pointMember = (y/mMember) - (bMemberQ/mMember)
                pointNonMember = (y/mNonMember) - (bNonMember/mNonMember) 
                ratio = pointMember/pointNonMember
                diff = pointMember - pointNonMember
            # Error cases
            elif x != None and y != None:
                print("Error: Both x and y values were provided. Please provide only one.")
                return (np.nan, np.nan)     
            else:
                print("Error: No point of comparison provided. Please provide an x or y value to test the ratio of.")
                return (np.nan, np.nan) 
            # Format plot
            plt.legend()
            # Construct lit comparison plot
            plt.figure()
            # Cooper+12 measured ~0.1 at redshift ~0.8
            plt.scatter(0.8, 0.1, label="Cooper+12")
            # Cooper+12 measured ~? at redshift ~1.3
            plt.scatter(1.3, 0.1, label="Raichoor+12")
            # Currently estimating our redshift to be at 1.25
            plt.scatter(1.25, diff, label="Our measurement")
            plt.xlabel("Redshift")
            plt.ylabel("Dlog Re (cluster - field)")
            plt.xlim(0, 2.5)
            plt.ylim(-0.2, 0.35)
            plt.legend()
            # Return our ratio and a nan value in the second slot in case this is incorrectly accessed
            return (ratio, np.nan)
        # Error case, for if an incorrect plotType is provided.
        print("Error: plot type not recognized. Please try again.")
        return (np.nan, np.nan)  
    # END COMPTRENDS

    def getBootstrapDiffs(self, region1:tuple=None, region2:tuple=None) -> list:
        """
        getBootstrapDiffs calculates the 68% confidence interval between two bootstrap uncertainty regions at a given index

        :param region1     :     Tuple containing 2 lists: 1) The list of x values covering the region and 2) the list of lists of y values delineating the uncertainty of all bootstrap lines at all x values
                                  Default: None
        :param region2     :     Tuple containing 2 lists: 1) The list of x values covering the region and 2) the list of lists of y values delineating the uncertainty of all bootstrap lines at all x values
                                  Default: None
        :return            :     3-Tuple containing 3 lists: 1) The list of lists of differences between all y values at all x indices, the list of upper confidence intervals at all x indices, 
                                  the list of lower confidence intervals at all x indices
        """
        # Initialize
        gridSize = len(region1[1]) # Same size as region1[0]
        lineCount = len(region1[1][0])
        diffs = np.empty((gridSize, lineCount))
        confUpper = np.empty((gridSize,))
        confLower = np.empty((gridSize,))
        # Construct 2D list of differences between all y values at all x indices
        for i in range(0, gridSize):
            for j in range(0, lineCount):
                diffs[i][j] = region1[1][i][j] - region2[1][i][j]
        # Calculate medians (not being used in current implementation)
        medDiffs = np.empty((gridSize,))
        for i in range(0, gridSize):
            medDiffs[i] = np.median(diffs[i])
        # Calculate 1D list of 68% confidence intervals
        for i in range(0, gridSize):
            confUpper[i] = np.percentile(diffs[i], 84)
            confLower[i] = np.percentile(diffs[i], 16)
        # Return 3-tuple of diffs and confidence intervals
        return diffs, confUpper, confLower
    #END GETBOOTSTRAPDIFFS

    def getMedian(self, category:str='SF', xRange:list=None, yRange:list=None):
        """
        Plots the median in four mass bins, including uncertainty and standard error on the median (THIS FUNCTION IS OUTDATED AND NEEDS TO BE REWORKED)

        :param category  :     Name of the category to consider when making comparisons
                                Default: SF
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
        plotStdError plots error bars for the standard error of a median
        
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
        Generates a table of slope and intercept values of best fit lines of all, passive, and star forming galaxies in each cluster  (THIS FUNCTION IS OUTDATED AND NEEDS TO BE REWORKED)

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
        
        :param : data - Pandas data frame correlating to one cluster. May be the entirety of the data for the cluster (after standard criteria have been applied)  (THIS FUNCTION IS OUTDATED AND NEEDS TO BE REWORKED)
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

    def testPlots(self, outputPath, truthPath):
        """
        Makes a series of test plots, stores and analyzes the data counts resulting from each plot to judge the accuracy of the plot() function.

        :param outputPath:  the file path that results will be written to.
        :param truthPath:  the file path that true values are stored in prior to calling, to be compared with results.

        :return:            test results are printed along with plots
        """
        # Establish criteria
        searchCriteria = [
            'Star == 0',
            'K_flag == 0',
            'Mstellar > 10**9.8',
            '(1 < zspec < 1.5) or ((((Redshift_Quality != 3) and (Redshift_Quality != 4)) or (SPECID < 0)) and (1 < zphot < 1.5))',
            'cluster_id <= 12',
            'totmask == 0',
            'Fit_flag > 2',
            'n < 6',
            'HSTFOV_flag == 1',
            're > 0'
        ]
        self.standardCriteria = searchCriteria

        with warnings.catch_warnings(): #suppressing depracation warnings for readability purposes
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
            # Open file for writing
            f = open(outputPath, 'w')
            # Establish variables for first test
            memberStatus = ["all", "only", "not"]
            plotType = [1, 2, 3]
            colorType = [None, "catalog", "passive", "sersic"]
            # Plot MSR and UVJ plots for each variable
            for m in memberStatus:
                for p in plotType:
                    for c in colorType:
                        if p == 1:
                            cluster = "SpARCS1616"
                        else:
                            cluster = None
                        xACountMSR, xBCountMSR, yACountMSR, yBCountMSR = self.plot('Mstellar', 're_converted', plotType=p, clusterName=cluster, useMembers=m, colorType=c, useLog=[True,True], xRange = [9.8, 11.5], yRange = [-0.5, 1.5], xLabel='log(Mstellar)', yLabel='log(Re)', fitLine=False)
                        xACountUVJ, xBCountUVJ, yACountUVJ, yBCountUVJ = self.plot('VMINJ', 'UMINV', plotType=p, clusterName=cluster, useMembers=m, colorType=c, useLog=[False,False], xRange = [-0.5,2.0], yRange = [0.0, 2.5], xLabel='V - J', yLabel='U - V', fitLine=False)
                        # End test early (and with specific error) if major discrepency is found
                        if xACountMSR != yACountMSR or xACountUVJ != yACountUVJ or xBCountMSR != yBCountMSR or xBCountUVJ != yBCountUVJ:
                            print("test failed. X and Y data counts do not agree.")
                            return
                        if xACountMSR != xACountUVJ:
                            print("test failed. stellar mass-size relation and UVJ counts do not agree.")
                            return
                        # Write A count
                        f.write('(' + str(xACountMSR) + ', ')
                        # Write B count
                        f.write(str(xBCountMSR) + ') ')
            # Establish variables for second test
            clusterNames = ["SpARCS0219", "SpARCS0035", "SpARCS1634", "SpARCS1616", "SPT0546", "SpARCS1638", "SPT0205", "SPT2106", "SpARCS1051", "SpARCS0335", "SpARCS1034"]
            xATot = 0
            xBTot = 0
            # Seperate results with newline
            f.write('\n')
            # Plot MSR plot for each cluster
            for cluster in clusterNames:
                xACount, xBCount, _, _ = self.plot('Mstellar', 're_converted', plotType=1, clusterName=cluster, useMembers="only", colorType=c, useLog=[True,True], xRange = [9.8, 11.5], yRange = [-0.5, 1.5], xLabel='log(Mstellar)', yLabel='log(Re)', fitLine=False)
                # Write A count
                f.write('(' + str(xACount) + ', ')
                # Write B count
                f.write(str(xBCount) + ') ')
                # Add value to total
                xATot+=xACount
                xBTot+=xBCount
            # Write total count on another newline
            f.write('\n(' + str(xATot) + ', ' + str(xBTot) + ') ')
            # Plot MSR plot for all clusters combined
            xATotExpected, xBTotExpected, _, _ = self.plot('Mstellar', 're_converted', plotType=3, clusterName=cluster, useMembers="only", colorType=c, useLog=[True,True], xRange = [9.8, 11.5], yRange = [-0.5, 1.5], xLabel='log(Mstellar)', yLabel='log(Re)', fitLine=False)
            # Write expected total
            f.write(str(xATotExpected))
            f.write(str(xBTotExpected))
            if xATot != xATotExpected or xBTot != xBTotExpected:
                print("test failed. Totaled Individual and combined cluster counts do not agree.")
                return
            # Establish variables for third test
            clusterNames = ["SpARCS0219", "SpARCS0035", "SpARCS1634", "SpARCS1616", "SPT0546", "SpARCS1638", "SPT0205", "SPT2106", "SpARCS1051", "SpARCS0335", "SpARCS1034"]
            xATot = 0
            xBTot = 0
            # Seperate results with newline
            f.write('\n')
            # Plot MSR plot for each cluster
            for cluster in clusterNames:
                xACount, xBCount, _, _ = self.plot('Mstellar', 're_converted', plotType=1, clusterName=cluster, useMembers="not", colorType=c, useLog=[True,True], xRange = [9.8, 11.5], yRange = [-1.5, 1.5], xLabel='log(Mstellar)', yLabel='log(Re)', fitLine=False)
                # Write A count
                f.write('(' + str(xACount) + ', ')
                # Write B count
                f.write(str(xBCount) + ') ')
                # Add value to total
                xATot+=xACount
                xBTot+=xBCount
            # Write total count on another newline
            f.write('\n(' + str(xATot) + ', ' + str(xBTot) + ') ')
            # Plot MSR plot for all clusters combined
            xATotExpected, xBTotExpected, _, _ = self.plot('Mstellar', 're_converted', plotType=3, clusterName=cluster, useMembers="not", colorType=c, useLog=[True,True], xRange = [9.8, 11.5], yRange = [-1.5, 1.5], xLabel='log(Mstellar)', yLabel='log(Re)', fitLine=False)
            # Write expected total
            f.write('(' + str(xATot) + ', ' + str(xBTot) + ') ')
            if xATot != xATotExpected or xBTot != xBTotExpected:
                print("test failed. Totaled Individual and combined field counts do not agree.")
                return
            f.close()
            # Open output and truth files for reading
            f = open(outputPath, 'r')
            testOutput = f.read()
            f.close()
            f = open(truthPath, 'r')
            expectedOutput = f.read()
            f.close()
            # Print result
            if testOutput == expectedOutput:
                print("test passed.")
                return
            print("test failed due to inconsistency with previous results.")
    # END TEST
    
    def plotUnwrapped(self, xQuantityName:str, yQuantityName:str, colorType:str=None, useLog:list=[False,False], fitLine:bool=False, additionalCriteria:list=None, useStandards:bool=False,
        color1:list=None, color2:list=None, plot=None, axes:list=None, row:int=None, col:int=None, bootstrap:bool=True, plotErrBars:bool=False, plotTransitionType:str=None):
            """
            Helper function called by plot(). Handles the plotting of data.
                
            :param xQuantityName:      Name of the column whose values are to be used as the x
            :param yQuantityName:      Name of the column whose values are to be used as the y
            :param colorType:          Specifies how to color code the plotted galaxies
                                        Default: None
                                        Value:   'catalog' - spectroscopic vs photometric catalog source
                                        Value:   'passive' - passive vs star forming
                                        Value:   'sersic' -  elliptical vs spiral
                                        Value:   'environment' - cluster vs field
                                        Value:   'environmentQ' - cluster vs field (Quiescent only)
                                        Value:   'environmentSF' - cluster vs field (Star-forming only)
            :param useLog:             Flag to indicate whether the x- or y-axis should be in log scale
                                        Default: [False,False] - neither axis in log scale
                                        Value:   [False,True] - y axis in log scale
                                        Value:   [True,False] - x axis in log scale
                                        Value:   [True,True] - both axis in log scale
            :param fitLine:            Flag to indicate whether a best fit line should be fit to the data. By default this line will plot size vs mass.
                                        Default: False 
            :param additionalCriteria: List of desired criteria the plotted galaxies should meet
                                        Default: None
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
            :param bootstrap:          Flag to indicate rather bootstrapping should be used to calculate and display uncertainty on the fit 
                                        Default: True
            :param plotErrBars:        Flag to indicate whether individual galaxies should have their Re error plotted (NOTE: this parameter assumes that you are plotting in log space)
                                        Default: False
            :param plotTransitionType: Allows for plotting a third category of data alongside two others (intended for use alongside 'passive' color type
                                        Default: None
                                        Value: GV - plot green valley trend
                                        Value: BQ - plot blue quiescent trend    
                                        Value: PSB - plot post-starburst trend   
            :return:                   (xA, xB, yA, yB), representing the total number of x-values and y-values corresponding to plotted data points of two different populations. Generated plot is displayed.
            """
            # Create 'goodData' flag for future checks
            self.setGoodData(additionalCriteria, useStandards)
            # Arbitrary establishment of variables for non-coloring case
            aData = self.catalog.query('goodData == 1')
            bData = self.catalog.query('goodData == 1')
            aLbl = None
            bLbl = None
            # Overwrite variables according to coloring scheme
            if colorType == None:
                # Don't need to do anything for this case. Included so program proceeds as normal
                pass
            elif colorType == 'catalog':
                aData = self.catalog.query('spectroscopic == 1 and goodData == 1')
                aLbl = 'Spectroscopic z'
                bData = self.catalog.query('photometric == 1 and goodData == 1')
                bLbl = 'Photometric z'
            elif colorType == 'passive':
                aData = self.catalog.query('passive == 1 and goodData == 1')
                aLbl = 'Quiescent'
                bData = self.catalog.query('starForming == 1 and goodData == 1')
                bLbl = 'Star Forming'
            elif colorType == 'GV':
                aData = self.catalog.query('greenValley == 1 and goodData == 1')
                aLbl = 'Green Valley'
                bData = self.catalog.query('greenValley == 0 and goodData == 1')
                bLbl = 'Other'
            elif colorType == 'BQ':
                aData = self.catalog.query('blueQuiescent == 1 and goodData == 1')
                aLbl = 'Blue Quiescent'
                bData = self.catalog.query('blueQuiescent == 0 and goodData == 1')
                bLbl = 'Other'
            elif colorType == 'PSB': 
                aData = self.catalog.query('postStarBurst == 1 and goodData == 1')
                aLbl = 'Post-starburst'
                bData = self.catalog.query('postStarBurst == 0 and goodData == 1')
                bLbl = 'Other' 
            elif colorType == 'sersic':
                aData = self.catalog.query('elliptical == 1 and goodData == 1')
                aLbl = 'Elliptical'
                bData = self.catalog.query('spiral == 1 and goodData == 1')
                bLbl = 'Spiral'
            elif colorType == 'environment':
                aData = self.catalog.query('member_adjusted == 1 and goodData == 1')
                aLbl = 'Cluster'
                bData = self.catalog.query('nonmember_adjusted == 1 and goodData == 1')
                bLbl = 'Field'
            elif colorType == 'environmentQ':
                aData = self.catalog.query('member_adjusted == 1 and passive == 1 and goodData == 1')
                aLbl = 'Cluster (Quiescent)'
                bData = self.catalog.query('nonmember_adjusted == 1 and passive == 1 and goodData == 1')
                bLbl = 'Field (Quiescent)'
            elif colorType == 'environmentSF':
                aData = self.catalog.query('member_adjusted == 1 and starForming == 1 and goodData == 1')
                aLbl = 'Cluster (Star-forming)'
                bData = self.catalog.query('nonmember_adjusted == 1 and starForming == 1 and goodData == 1')
                bLbl = 'Field (Star-forming)'
            else:
                print(colorType, ' is not a valid coloring scheme!')
                return
            # Code for plotting transition population trends, like in Matharu+20
            if plotTransitionType == 'GV':
                cData = self.catalog.query('greenValley == 1 and goodData == 1')
                cLbl = 'Green Valley'
                cColor = "green"
            elif plotTransitionType == 'BQ':
                cData = self.catalog.query('blueQuiescent == 1 and goodData == 1')
                cLbl = 'Blue Quiescent'
                cColor = "black"
            elif plotTransitionType == 'PSB':
                cData = self.catalog.query('postStarBurst == 1 and goodData == 1')
                cLbl = 'Post-starburst'
                cColor = "black"
            aXVals = aData[xQuantityName].values
            bXVals = bData[xQuantityName].values
            aYVals = aData[yQuantityName].values
            bYVals = bData[yQuantityName].values
            # Check if either axis needs to be converted to log space for plotting purposes
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
                    self.MSRfit(aData, useLog, axes, row, col, bootstrap=bootstrap)
            # Generate the plot
            plot.scatter(aXVals, aYVals, alpha=0.5, color=color1, label=aLbl)
            # Generate a third line and color in transition data if plotting a transition type
            if plotTransitionType != None:
                self.MSRfit(cData, useLog, axes, row, col, typeRestrict=cLbl, color=cColor, bootstrap=bootstrap)
                cXVals = cData[xQuantityName].values
                cYVals = cData[yQuantityName].values
                # Check if either axis needs to be converted to log space for plotting purposes
                if useLog[0] == True:
                    cXVals = np.log10(cXVals)
                if useLog[1] == True:
                    cYVals = np.log10(cYVals)
                plot.scatter(cXVals, cYVals, alpha=0.5, s=70, marker=mrk.MarkerStyle(marker='s', fillstyle='none'), color=cColor)
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
                    # Convert to log space
                    upperSigma = np.log10(pow(10, size) + sigma) - np.log10(pow(10, size))
                    lowerSigma = np.log10(pow(10, size)) - np.log10(pow(10, size) - sigma)
                    if np.isnan(upperSigma) or np.isnan(lowerSigma):
                        plt.scatter(mass, size, alpha=0.5, color='black')
                    else:
                        plt.errorbar(mass, size, upperSigma, barsabove = True, ecolor=color1)
                        plt.errorbar(mass, size, lowerSigma, barsabove = False, ecolor=color1)
                for i in range(0, len(bXVals)):
                    mass = bXVals[i]
                    size = bYVals[i]
                    sigma = bYsigmas[i]
                    # Convert to log space
                    upperSigma = np.log10(pow(10, size) + sigma) - np.log10(pow(10, size))
                    lowerSigma = np.log10(pow(10, size)) - np.log10(pow(10, size) - sigma)
                    if np.isnan(upperSigma) or np.isnan(lowerSigma):
                        plt.scatter(mass, size, alpha=0.5, color='black')
                    else:
                        plt.errorbar(mass, size, upperSigma, barsabove = True, ecolor=color2)
                        plt.errorbar(mass, size, lowerSigma, barsabove = False, ecolor=color2)
            if colorType != None:
                plot.scatter(bXVals, bYVals, alpha=0.5, color=color2, label=bLbl)
            # Plot van der Wel et al. 2014 line in the case where we are plotting the stellar mass-size relation (passive v starforming) for the field.
            if xQuantityName == 'Mstellar' and (yQuantityName == 're' or yQuantityName == 're_converted') and colorType == "passive" and ("nonmember_adjusted == 1" in additionalCriteria or "nonmember_adjusted == 1" in self.standardCriteria):
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
            return (xA, xB, yA, yB)
    # END PLOTUNWRAPPED


    def plot(self, xQuantityName:str, yQuantityName:str, plotType:int, clusterName:str=None, useMembers:str='all', colorType:str=None, colors:list=None, 
        useStandards:bool=True, xRange:list=None, yRange:list=None, xLabel:str='', yLabel:str='', useLog:list=[False,False], fitLine:bool=False, bootstrap:bool=True, plotErrBars:bool=False, plotTransitionType:str=None):
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
        :param useMembers:        Flag to indicate whether only cluster members should be plotted or only non-members should be plotted.
                                    Default: 'only' - only members
                                    Value:   'not' - only non-members
                                    Value:   'all' - no restriction imposed
        :param colorType:          Specifies how to color code the plotted galaxies
                                    Default: None
                                    Value:   'catalog' - spectroscopic vs photometric catalog source
                                    Value:   'passive' - passive vs star forming
                                    Value:   'sersic' -  elliptical vs spiral
                                    Value:   'environment' - cluster vs field
        :param colors:             Specifies what colors should be used when plotting
                                    Default: None - default colors are used
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
        :param fitLine:            Flag to indicate whether a best fit line should be fit to the data. By default this line will plot size vs mass. 
                                    Default: False
        :param bootstrap:          Flag to indicate rather bootstrapping should be used to calculate and display uncertainty on the fit 
                                    Default: True
        :param plotErrBars:        Flag to indicate whether individual galaxies should have their Re error plotted
                                    Default: False
        :param plotTransitionType: Allows for plotting a third category of data alongside two others (intended for use alongside 'passive' color type
                                    Default: None
                                    Value: GV - plot green valley trend
                                    Value: BQ - plot blue quiescent trend    
                                    Value: PSB - plot post-starburst trend                     
        :return:                   (xA, xB, yA, yB), representing the total number of x-values and y-values corresponding to plotted data points in two populations. Generated plot is displayed.
        """
        # Initialize additional criteria
        additionalCriteria = []
        # Initialize plot
        plt.figure(figsize=(8,6))
        # Check if plot colors were provided by the user
        if colors != None:
            color1 = colors[0]
            color2 = colors[1]
        # If not, generate default colors
        else:
            if colorType == 'passive':
                color1 = "red"
                color2 = "blue"
            else:
                color1 = "green"
                color2 = "orange"
        # Establish membership criteria
        if useMembers == None:
            print("Please specify membership requirements!")
            return
        elif useMembers == 'all':
            # Reduce data to only contain galaxies classified as members or non-members
            additionalCriteria.append('member_adjusted == 1 or nonmember_adjusted == 1')
        elif useMembers == 'only':
            # Reduce data to only contain galaxies classified as members
            additionalCriteria.append('member_adjusted == 1')
        elif useMembers == 'not':
            # Reduce data to only contain galaxies not classified as members
            additionalCriteria.append('nonmember_adjusted == 1')
        else:
            print(useMembers, " is not a valid membership requirement!")
            return
        # Plot only the cluster specified
        if plotType == 1:
            if clusterName == None:
                print("No cluster name provided!")
                return
            # Plot data
            clusterCriterion = 'cluster == \'' + clusterName + '\''
            additionalCriteria.append(clusterCriterion)
            xATot, xBTot, yATot, yBTot = self.plotUnwrapped(xQuantityName=xQuantityName, yQuantityName=yQuantityName, colorType=colorType, useLog=useLog, fitLine=fitLine, additionalCriteria=additionalCriteria, 
                useStandards=useStandards, color1=color1, color2=color2, plot=plt, bootstrap=bootstrap, plotErrBars=plotErrBars, plotTransitionType=plotTransitionType)
        # Plot all clusters individually in a subplot
        elif plotType == 2:
            # Initialize data count totals (used when running test suite)
            xATot = 0
            xBTot = 0
            yATot = 0
            yBTot = 0
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
                    # Plot data for current cluster
                    currentAdditionalCriteria = additionalCriteria.copy()
                    currentAdditionalCriteria.append('cluster == \'' + currentClusterName + '\'')
                    xA, xB, yA, yB = self.plotUnwrapped(xQuantityName=xQuantityName, yQuantityName=yQuantityName, colorType=colorType, useLog=useLog, fitLine=fitLine, additionalCriteria=currentAdditionalCriteria, 
                        useStandards=useStandards, color1=color1, color2=color2, plot=axes[i][j], axes=axes, row=i, col=j, bootstrap=bootstrap, plotErrBars=plotErrBars, plotTransitionType=plotTransitionType)
                    # Update data count totals
                    xATot+=xA
                    xBTot+=xB
                    yATot+=yA
                    yBTot+=yB
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
            xATot, xBTot, yATot, yBTot = self.plotUnwrapped(xQuantityName=xQuantityName, yQuantityName=yQuantityName, colorType=colorType, useLog=useLog, fitLine=fitLine, additionalCriteria=additionalCriteria, 
                useStandards=useStandards, color1=color1, color2=color2, plot=plt, bootstrap=bootstrap, plotErrBars=plotErrBars, plotTransitionType=plotTransitionType)
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
        # Return
        return (xATot, xBTot, yATot, yBTot)
    # END PLOT