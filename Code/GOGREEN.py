from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rng
import os


"""
Freeman's to-do:
1. Improve file format used by makeTable()
2. draw the quiescent border line in 'passive' colorType plots.
3. Manage fit-line color to be more helpful
"""
class GOGREEN:
    def __init__(self, dataPath:str):
        """
        __init__ Constructor to define and initialize class members

        :param dataPath: absolute path to the directory containing DR1/ and STRUCTURAL_PARA_v1.1_CATONLY/
                         subdirectories
        """ 
        
        self.catalog = pd.DataFrame()
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

        # Build path string to the photometric catalog
        photoCatPath = self._path + 'DR1/CATS/Photo.fits'
        # Generate a DataFrame of the catalog data
        self._photoCatalog = self.generateDF(photoCatPath)

        # Build path string to the redshift catalog
        redshiftCatPath = self._path + 'DR1/CATS/Redshift_catalogue.fits'
        # Generate a DataFrame of the catalog data
        self._redshiftCatalog = self.generateDF(redshiftCatPath)

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
    # END INIT

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

    def getMembers(self, clusterName:str) -> pd.DataFrame:
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
        specZthreshold = np.abs(allClusterGalaxies['zspec'].values-clusterZ) < 0.02*(1+allClusterGalaxies['zspec'].values)
        specZgalaxies = allClusterGalaxies[specZthreshold]
        # Photometric criteria: (zphot-zclust) < 0.08(1+zphot)
        photZthreshold = np.abs(allClusterGalaxies['zphot'].values-clusterZ) < 0.08*(1+allClusterGalaxies['zphot'].values)
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
        clusterZ = self.getClusterZ(clusterName)
        allClusterGalaxies = self.getClusterGalaxies(clusterName)
        # Find spectroscopic and photometric non-members seperately
        # Spectrosocpic criteria: (zspec-zclust) > 0.02(1+zspec)
        specZthreshold = np.abs(allClusterGalaxies['zspec'].values-clusterZ) > 0.02*(1+allClusterGalaxies['zspec'].values)
        specZgalaxies = allClusterGalaxies[specZthreshold]
        # Photometric criteria: (zphot-zclust) > 0.08(1+zphot)
        photZthreshold = np.abs(allClusterGalaxies['zphot'].values-clusterZ) > 0.08*(1+allClusterGalaxies['zphot'].values)
        photZgalaxies = allClusterGalaxies[photZthreshold]
        # Remove photZgalaxies with a specZ
        photZgalaxies = photZgalaxies[~photZgalaxies['cPHOTID'].isin(specZgalaxies['cPHOTID'])]
        # Combine into a single DataFrame
        memberGalaxies = specZgalaxies.append(photZgalaxies)
        return memberGalaxies
    # END GETNONMEMBERS

    def reduceDF(self, frame:pd.DataFrame, additionalCriteria:list, useStandards:bool) -> pd.DataFrame:
        """
        reduceDF Reduces the DataFrame param:frame to contain only galaxies that meet the criteria provided in
                 param:additionalCriteria and the standard criteria (if param:useStandards is True)

        :param additionalCriteria: List of criteria to apply to param:frame
        :param useStandards:       Flag to specify whether the standard criteria should be applied to param:frame
        :return:                   Pandas DataFrame containing the galaxies whose values meet the criteria within param:additionalCriteria
                                   and the standard criteria (if param:useStandards is True)
        """
        if (additionalCriteria != None):
            for criteria in additionalCriteria:
                frame = frame.query(criteria)
        if useStandards:
            for criteria in self.standardCriteria:
                frame = frame.query(criteria)
        return frame
    # END REDUCEDF
        
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
        logA = 0.7
        alpha = 0.22
        Mstellar = np.array([3162280000, 316228000000])
        Re = np.array([-1.5, 1.5])
        # In case of subplots, plot for the specific row and column
        if row != None and col != None:
            if axes[row][col] != None:
                axes[row][col].plot(Re, logA + (alpha * np.log(Mstellar / (5 * np.power(10, 10)))), linestyle='dashed', color='black')
                print("got lines")
                return
        # Else plot normally
        plt.plot(Re, logA + (alpha * np.log(Mstellar / (5 * np.power(10, 10)))), linestyle='dashed', color='black')
        print("got lines")
    #END PLOTVANDERWELLINES

    def reConvert(self, data:list) -> list:
        """
        reConvert (private method) convert effective radius values from units of arcsec to kpc.

        :param data:   The set of data being used by the calling function, plot().
        :return   :    returns the list of converted effective radius values

        """
        sizes = data['re'].values * (cosmo.kpc_proper_per_arcmin(data['zspec'].values)/60) #converting all effective radii from units of arcsec to kpc using their spectroscopic redshifts
        for i in range(0, len(sizes)):
            if np.isnan(sizes[i]) == True: #checking where conversion failed due to lack of zspec value
                sizes[i] = data['re'].values[i] * (cosmo.kpc_proper_per_arcmin(data['zphot'].values[i])/60) #use photometric redshifts instead where there are no spectroscopic redshifts
            if np.isnan(sizes[i]) == True: #checking if there are any remaining missing values (either because there is no redshift or there is no re)
                sizes[i] = 1 #setting to arbitrary value so polyfit will not fail (is there a way to simply exclude these galaxies?).  NOTE: arbitrary value CANNOT be 0.
        sizes = (sizes / u.kpc) * u.arcmin # removing units so the data can be used in the functions below
        return sizes
    #END RECONVERT

    def MSRfit(self, data:list, useLog:list=[False, False], axes:list=None, row:int=None, col:int=None, allData:bool=False, useMembers:str='only', additionalCriteria:list=None, useStandards:bool=True, typeRestrict:str=None, color:str='black'):
        """
        MSRfit (private method) fit a best fit line to data generated by the plot() method

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
        :param allData:             Flag to indicate whether all cluster data needs to be compiled in place of the data parameter.
                                     Default: False
        :param useMembers:        Flag to indicate whether only cluster members should be plotted or only non-members should be plotted.
                                    Default: 'only' - only members
                                    Value:   'not' - only non-members
                                    Value:   'all' - no restriction imposed
        :param additionalCriteria:  List of desired criteria the plotted galaxies should meet
                                     Default: None
        :param useStandards:        Flag to indicate whether the standard search criteria should be applied
                                     Default: True
        :param typeRestrict:        Flag to indicate whether data should be restricted based on SFR (only necessary when allData is True)
                                     Default: None
                                     Value:   'passive' - only passive galaxies should be considered.
                                     Value:   'starForming' - only star forming galaxies should be considered.
                                     Value:   'elliptical' - only galaxies with 2.5 < n < 6 should be considered.
                                     Value:   'spiral' - only galaxies with n < 2.5 should be considered.
        :param color1:             The color the fit line should be.
                                    Default: 'black'       
        :return   :

        """
        # If this function is called in a situation where galaxies in all clusters should be considered, it will handle the process of concatenating all of the data together prior to fitting.
        # In this situation, the input value for 'data' is ignored.
        if allData:
            # Set an initial value to append to.
            data = pd.DataFrame()
            for clusterName  in self._structClusterNames:
                if useMembers == 'only':
                    # Get data for this cluster for galaxies classified as members
                    data = data.append(self.getMembers(clusterName))
                elif useMembers == 'not':
                    # Get data for this cluster for galaxies classified as non-members
                    data = data.append(self.getNonMembers(clusterName))
                else:
                    # Get data for this cluster for all galaxies
                    data = data.append(self.getClusterGalaxies(clusterName))
            # Apply other specified reducing constraints
            data = self.reduceDF(data, additionalCriteria, useStandards)
            # Handle case where only passive galaxies out of all data need to plotted.
            if typeRestrict == 'passive':
                data = data.query('(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)')
            # Handling case where only star forming galaxies out of all data need to plotted.
            if typeRestrict == 'starForming':
                data = data.query('(UMINV <= 1.3) or (VMINJ >= 1.6) or (UMINV <= 0.60+VMINJ)')
            if typeRestrict == 'elliptical':
                data = data.query('2.5 < n < 6')
            # Handling case where only star forming galaxies out of all data need to plotted.
            if typeRestrict == 'spiral':
                data = data.query('n < 2.5')

        # Convert all effective radii from units of arcsec to kpc using their spectroscopic redshifts
        size = self.reConvert(data)

        mass = data['Mstellar'].values
        badDataIndices = []
        for i in range(0, len(mass)):
            # Check if there are any remaining missing values (in the rare case where there is no Mstellar value)
            if np.isnan(mass[i]) == True:
                # Add the index of this data point to the list of those to be removed once all data points have been checked.
                badDataIndices.append(i)
        for j in range(0, len(badDataIndices)):
            # Iterate through the array of indices, removing the data at these indices from both axis arrays.
            mass = np.delete(mass, badDataIndices[j])
            size = np.delete(size, badDataIndices[j])
            # Adjusting badDataIndices to account for reduced count of all further recorded indices
            for k in range(0, len(badDataIndices)):
                badDataIndices[k] = badDataIndices[k] - 1


        xFitData = mass
        yFitData = size
        if useLog[0] == True:
            xFitData = np.log10(xFitData)
        if useLog[1] == True:
            yFitData = np.log10(yFitData)
        m, b = np.polyfit(xFitData, yFitData, 1) #slope and intercept for best fit line
        if row != None and col != None:
            # Check for subplots
            if axes[row][col] != None:
                # Add white backline in case of plotting multiple fit lines in one plot
                if color != 'black':
                    axes[row][col].plot(xFitData, m * xFitData + b, color='white', linewidth=4)
                # Plot the best fit line
                axes[row][col].plot(xFitData, m * xFitData + b, color=color)
                return
        # Add white backline in case of plotting multiple fit lines in one plot
        if color != 'black':
            plt.plot(xFitData, m * xFitData + b, color='white', linewidth=4)
        # Plot the best fit line
        plt.plot(xFitData, m * xFitData + b, color=color)
    # END MSRFIT

    def getRatio(self, category:str='SF', x:float=None, y:float=None, plotLines:bool=False, xRange:list=None, yRange:list=None) -> list:
        """
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

            # Convert all effective radii from units of arcsec to kpc using their spectroscopic redshifts
            memberSizeQ = self.reConvert(memberDataQ)
            memberSizeSF = self.reConvert(memberDataSF)
            nonMemberSizeQ = self.reConvert(nonMemberDataQ)
            nonMemberSizeSF = self.reConvert(nonMemberDataSF)

            memberMassQ = memberDataQ['Mstellar'].values
            memberMassSF = memberDataSF['Mstellar'].values
            nonMemberMassQ = nonMemberDataQ['Mstellar'].values
            nonMemberMassSF = nonMemberDataSF['Mstellar'].values
            badDataIndices = []
            for i in range(0, len(memberMassQ)):
                # Check if there are any remaining missing values (in the rare case where there is no Mstellar value)
                if np.isnan(memberMassQ[i]) == True:
                    # Add the index of this data point to the list of those to be removed once all data points have been checked.
                    badDataIndices.append(i)
            for j in range(0, len(badDataIndices)):
                # Iterate through the array of indices, removing the data at these indices from both axis arrays.
                memberMassQ = np.delete(memberMassQ, badDataIndices[j])
                memberSizeQ = np.delete(memberSizeQ, badDataIndices[j])
                # Adjusting badDataIndices to account for reduced count of all further recorded indices
                for k in range(0, len(badDataIndices)):
                    badDataIndices[k] = badDataIndices[k] - 1
            badDataIndices = []
            for i in range(0, len(memberMassSF)):
                # Check if there are any remaining missing values (in the rare case where there is no Mstellar value)
                if np.isnan(memberMassSF[i]) == True:
                    # Add the index of this data point to the list of those to be removed once all data points have been checked.
                    badDataIndices.append(i)
            for j in range(0, len(badDataIndices)):
                # Iterate through the array of indices, removing the data at these indices from both axis arrays.
                memberMassSF = np.delete(memberMassSF, badDataIndices[j])
                memberSizeSF = np.delete(memberSizeSF, badDataIndices[j])
                # Adjusting badDataIndices to account for reduced count of all further recorded indices
                for k in range(0, len(badDataIndices)):
                    badDataIndices[k] = badDataIndices[k] - 1
            badDataIndices = []
            for i in range(0, len(nonMemberMassQ)):
                # Check if there are any remaining missing values (in the rare case where there is no Mstellar value)
                if np.isnan(nonMemberMassQ[i]) == True:
                    # Add the index of this data point to the list of those to be removed once all data points have been checked.
                    badDataIndices.append(i)
            for j in range(0, len(badDataIndices)):
                # Iterate through the array of indices, removing the data at these indices from both axis arrays.
                nonMemberMassQ = np.delete(nonMemberMassQ, badDataIndices[j])
                nonMemberSizeQ = np.delete(nonMemberSizeQ, badDataIndices[j])
                # Adjusting badDataIndices to account for reduced count of all further recorded indices
                for k in range(0, len(badDataIndices)):
                    badDataIndices[k] = badDataIndices[k] - 1
            badDataIndices = []
            for i in range(0, len(nonMemberMassSF)):
                # Check if there are any remaining missing values (in the rare case where there is no Mstellar value)
                if np.isnan(nonMemberMassSF[i]) == True:
                    # Add the index of this data point to the list of those to be removed once all data points have been checked.
                    badDataIndices.append(i)
            for j in range(0, len(badDataIndices)):
                # Iterate through the array of indices, removing the data at these indices from both axis arrays.
                nonMemberMassSF = np.delete(nonMemberMassSF, badDataIndices[j])
                nonMemberSizeSF = np.delete(nonMemberSizeSF, badDataIndices[j])
                # Adjusting badDataIndices to account for reduced count of all further recorded indices
                for k in range(0, len(badDataIndices)):
                    badDataIndices[k] = badDataIndices[k] - 1
            memberMassQ = np.log10(memberMassQ)
            memberSizeQ = np.log10(memberSizeQ)
            memberMassSF = np.log10(memberMassSF)
            memberSizeSF = np.log10(memberSizeSF)
            nonMemberMassQ = np.log10(nonMemberMassQ)
            nonMemberSizeQ = np.log10(nonMemberSizeQ)
            nonMemberMassSF = np.log10(nonMemberMassSF)
            nonMemberSizeSF = np.log10(nonMemberSizeSF)
            mMemberQ, bMemberQ = np.polyfit(memberMassQ, memberSizeQ, 1)
            mMemberSF, bMemberSF = np.polyfit(memberMassSF, memberSizeSF, 1)
            mNonMemberQ, bNonMemberQ = np.polyfit(nonMemberMassQ, nonMemberSizeQ, 1)
            mNonMemberSF, bNonMemberSF = np.polyfit(nonMemberMassSF, nonMemberSizeSF, 1)
            if plotLines:
                plt.plot(memberMassQ, mMemberQ * memberMassQ + bMemberQ, label='quiescent members')
                plt.plot(memberMassSF, mMemberSF * memberMassSF + bMemberSF, label='star-forming members')
                plt.plot(nonMemberMassQ, mNonMemberQ * nonMemberMassQ + bNonMemberQ, label='quiescent non-members')
                plt.plot(nonMemberMassSF, mNonMemberSF * nonMemberMassSF + bNonMemberSF, label='star-forming non-members')
                plt.legend()
                if xRange != None:
                    if len(xRange) > 1:
                        plt.xlim(xRange[0], xRange[1])
                if yRange != None:
                    if len(yRange) > 1:
                        plt.ylim(yRange[0], yRange[1])
                plt.xlabel('log(Mstellar)')
                plt.ylabel('log(Re)')
            if x != None:
                pointMemberQ = x*mMemberQ + bMemberQ
                pointMemberSF = x*mMemberSF + bMemberSF 
                pointNonMemberQ = x*mNonMemberQ + bNonMemberQ 
                pointNonMemberSF = x*mNonMemberSF + bNonMemberSF
                ratioQ = pointMemberQ/pointNonMemberQ
                ratioSF = pointMemberSF/pointNonMemberSF
                return [ratioQ, ratioSF]
            elif y != None:
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
        else:
            print(category + " is not a valid category")
            return [-1]
    #END GETRATIO

    def getMedian(self, category:str='SF', xRange:list=None, yRange:list=None, printError:bool=False):
        """
        :param category  :     Name of the category to consider when making comparisons
        :param xRange    :     List containing the desired lower and upper bounds for the x-axis
                                Default: None
        :param yRange    :     List containing the desired lower and upper bounds for the y-axis
                                Default: None
        :param printError:     flag indicating whether or not standard error values should printed alongside the plot
                                Default: False

        :return: medians are plotted
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
            # Plot the 4 sets of medians across 4 mass bins
            xValues = np.array([9.75, 10.25, 10.75, 11.25])
            offsetMQ = 0
            offsetMSF = 0.03
            offsetNMQ = -0.03
            offsetNMSF = 0.06
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
            # Calculate standard error for each of the 16 medians if printError is true
            if printError:
                stdErrorsMQ = np.array([self.getStdError(i) for i in sizeMQ])
                stdErrorsMSF = np.array([self.getStdError(i) for i in sizeMSF])
                stdErrorsNMQ = np.array([self.getStdError(i) for i in sizeNMQ])
                stdErrorsNMSF = np.array([self.getStdError(i) for i in sizeNMSF])
                for i in range(0, len(xValues)):
                    self.plotStdError(medianMQ[i], xValues[i] + offsetMQ, stdErrorsMQ[i], 'black')
                    self.plotStdError(medianMSF[i], xValues[i] + offsetMSF, stdErrorsMSF[i], 'black')
                    self.plotStdError(medianNMQ[i], xValues[i] + offsetNMQ, stdErrorsNMQ[i], 'black')
                    self.plotStdError(medianNMSF[i], xValues[i] + offsetNMSF, stdErrorsNMSF[i], 'black')
                # NOTE: size (dy) of upper and lower error bar will be asymetric because it is LOG
    #END GETMEDIAN

    def getStdError(self, data:list=None) -> int:
        return 1.253 * (np.std(data)/np.sqrt(len(data)))
    #END GETSTDERROR

    def plotStdError(self, median:int=None, bin:int=None, stdError:int=None, color:str=None):
        plt.errorbar(bin, median, stdError, barsabove = True, ecolor=color)
        plt.errorbar(bin, median, stdError, barsabove = False, ecolor=color)
    #END PLOTSTDERROR

    def plotUncertainties(self, data:list=None, median:int=None, bin:int=None, color:str=None):
        confLower = np.percentile(data, 25)
        confHigher = np.percentile(data, 75)
        plt.errorbar(bin, median, confHigher - median, barsabove=True, ecolor=color)
        plt.errorbar(bin, median, median - confLower, barsabove=False, ecolor=color)
    #END PLOTUNCERTAINTY

    def makeTable(self, filename):
        """
        :param : filename - the name of the file to write to.
        :return: writes slope and y-intercept of best fit lines of all, passive, and star forming galaxies in each cluster to the file 'output.txt' (better file type to be implemented in the future)
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

        mass = data['Mstellar'].values
        badDataIndices = []
        for i in range(0, len(mass)):
            # Check if there are any remaining missing values (in the rare case where there is no Mstellar value)
            if np.isnan(mass[i]) == True:
                # Add the index of this data point to the list of those to be removed once all data points have been checked.
                badDataIndices.append(i)
        for j in range(0, len(badDataIndices)):
            # Iterate through the array of indices, removing the data at these indices from both axis arrays.
            mass = np.delete(mass, badDataIndices[j])
            size = np.delete(size, badDataIndices[j])
            # Adjusting badDataIndices to account for reduced count of all further recorded indices
            for k in range(0, len(badDataIndices)):
                badDataIndices[k] = badDataIndices[k] - 1
        
        xFitData = np.log10(mass)
        yFitData = np.log10(size)
        m, b = np.polyfit(xFitData, yFitData, 1) #slope and intercept for best fit line
        f.write(str(m) + ' ')
        f.write(str(b) + ' ')    
    # END MAKETABLE

    def plot(self, xQuantityName:str, yQuantityName:str, plotType:int, clusterName:str=None, additionalCriteria:list=None, useMembers:str='only', colorType:str=None,
             colors:list=None, useStandards:bool=True, xRange:list=None, yRange:list=None, xLabel:str='', yLabel:str='', useLog:list=[False,False], fitLine:bool=False):
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
        :return:                  The generated plot(s) will be displayed
        """
        # Generate random colors
        color1 = [1, rng.random(), rng.random()]
        color2 = [0, rng.random(), rng.random()]
        # Check if plot colors were provided by the user
        if (colors != None):
            color1 = colors[0]
            color2 = colors[1]
        # Plot only the cluster specified
        if plotType == 1:
            if clusterName == None:
                print("No cluster name provided!")
                return
            # Get all galaxies associated with this cluster
            data = self.getClusterGalaxies(clusterName)
            if useMembers == 'only':
                # Reduce data to only contain galaxies classified as members
                data = self.getMembers(clusterName)
            if useMembers == 'not':
                # Reduce data to only contain galaxies not classified as members
                data = self.getNonMembers(clusterName)
            # Apply other specified reducing constraints
            data = self.reduceDF(data, additionalCriteria, useStandards)
            # Plot depending on how the values should be colored
            if colorType == None:
                # Extract desired quantities from data
                xData = data[xQuantityName].values
                yData = data[yQuantityName].values
                # Check if either axis is measuring effective radius for the purpose of unit conversion.
                if xQuantityName == 're':
                    xData = self.reConvert(data)
                if yQuantityName == 're':
                    yData = self.reConvert(data)
                # Check if either axis needs to be put in log scale
                if useLog[0] == True:
                    xData = np.log10(xData)
                if useLog[1] == True:
                    yData = np.log10(yData)
                # Generate the plot
                plt.scatter(xData, yData, color=color1)
                # generate best fit line
                if fitLine == True:
                    self.MSRfit(data, useLog)
                # Plot van der Wel et al. 2014 line in the case where we are plotting MSR
                if xQuantityName == 'Mstellar' and yQuantityName == 're':
                    self.plotVanDerWelLines()
            elif colorType == 'membership':
                # Extract desired quantities from data
                specZ = data[~data['zspec'].isna()]
                # Assume photZ are those that do not have a specZ
                photZ = data[~data['cPHOTID'].isin(specZ['cPHOTID'])]
                specXData = specZ[xQuantityName].values
                specYData = specZ[yQuantityName].values
                photXData = photZ[xQuantityName].values
                photYData = photZ[yQuantityName].values
                # Check if either axis is measuring effective radius for the purpose of unit conversion.
                if xQuantityName == 're':
                    specXData = self.reConvert(specZ)
                    photXData = self.reConvert(photZ)
                if yQuantityName == 're':
                    specYData = self.reConvert(specZ)
                    photYData = self.reConvert(photZ)
                # Check if either axis needs to be put in log scale
                if useLog[0] == True:
                    specXData = np.log10(specXData)
                    photXData = np.log10(photXData)
                if useLog[1] == True:
                    specYData = np.log10(specYData)
                    photYData = np.log10(photYData)
                # Generate the plot
                plt.scatter(specXData, specYData, color=color1, label='Spectroscopic z')
                plt.scatter(photXData, photYData, color=color2, label='Photometric z')
                # generate best fit line
                if fitLine == True:
                    self.MSRfit(data, useLog)
                # Plot van der Wel et al. 2014 line in the case where we are plotting MSR
                if xQuantityName == 'Mstellar' and yQuantityName == 're':
                    self.plotVanDerWelLines()
            elif colorType == 'passive':
                # Build passive query string (from van der Burg et al. 2020), limiting mass to > 10^9.7
                passiveQuery = '(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ) and Mstellar > 5011870000'
                # Build active query string, limiting mass to > 10^9.5
                starFormingQuery = '(UMINV <= 1.3) and (VMINJ <= 1.6) and (UMINV <= 0.60+VMINJ) and Mstellar > 3162280000'
                # Extract desired quantities from data
                passive = data.query(passiveQuery)
                starForming = data.query(starFormingQuery)
                passiveX = passive[xQuantityName].values
                passiveY = passive[yQuantityName].values
                starFormingX = starForming[xQuantityName].values
                starFormingY = starForming[yQuantityName].values
                # Check if either axis is measuring effective radius for the purpose of unit conversion.
                if xQuantityName == 're':
                    passiveX = self.reConvert(passive)
                    starFormingX = self.reConvert(starForming)
                if yQuantityName == 're':
                    passiveY = self.reConvert(passive)
                    starFormingY = self.reConvert(starForming)
                # Check if either axis needs to be put in log scale
                if useLog[0] == True:
                    passiveX = np.log10(passiveX)
                    starFormingX = np.log10(starFormingX)
                if useLog[1] == True:
                    passiveY = np.log10(passiveY)
                    starFormingY = np.log10(starFormingY)
                # Generate the plot
                plt.scatter(passiveX, passiveY, color=color1, label='Quiescent')
                plt.scatter(starFormingX, starFormingY, color=color2, label='Star Forming')
                # generate best fit line
                if fitLine == True:
                    self.MSRfit(passive, useLog, color=color1)
                    self.MSRfit(starForming, useLog, color=color2)
                # Plot passive v star-forming border in the case where we are plotting UVJ color-color
                if xQuantityName == 'VMINJ' and yQuantityName == 'UMINV':
                    self.plotPassiveLines()
                # Plot van der Wel et al. 2014 line in the case where we are plotting MSR
                if xQuantityName == 'Mstellar' and yQuantityName == 're':
                    self.plotVanDerWelLines()
            elif colorType == 'sersic':
                elliptical = data.query('2.5 < n < 6')
                spiral = data.query('n < 2.5')
                ellipticalX = elliptical[xQuantityName].values
                ellipticalY = elliptical[yQuantityName].values
                spiralX = spiral[xQuantityName].values
                spiralY = spiral[yQuantityName].values
                # Check if either axis is measuring effective radius for the purpose of unit conversion.
                if xQuantityName == 're':
                    ellipticalX = self.reConvert(elliptical)
                    spiralX = self.reConvert(spiral)
                if yQuantityName == 're':
                    ellipticalY = self.reConvert(elliptical)
                    spiralY = self.reConvert(spiral)
                # Check if either axis needs to be put in log scale
                if useLog[0] == True:
                    ellipticalX = np.log10(ellipticalX)
                    spiralX = np.log10(spiralX)
                if useLog[1] == True:
                    ellipticalY = np.log10(ellipticalY)
                    spiralY = np.log10(spiralY)
                # Generate the plot
                plt.scatter(ellipticalX, ellipticalY, color=color1, label='2.5 < n < 6')
                plt.scatter(spiralX, spiralY, color=color2, label='n < 2.5')
                # generate best fit line
                if fitLine == True:
                    self.MSRfit(elliptical, useLog, color=color1)
                    self.MSRfit(spiral, useLog, color=color2)
                # Plot van der Wel et al. 2014 line in the case where we are plotting MSR
                if xQuantityName == 'Mstellar' and yQuantityName == 're':
                    self.plotVanDerWelLines()
            else:
                print(colorType, ' is not a valid coloring scheme!')

        # Plot all clusters individually in a subplot
        elif plotType == 2:
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
                    # Get all galaxies associated with this cluster
                    data = self.getClusterGalaxies(currentClusterName)
                    if useMembers == 'only':
                        # Reduce data to only contain galaxies classified as members
                        data = self.getMembers(currentClusterName)
                    if useMembers == 'not':
                        # Reduce data to only contain galaxies not classified as members
                        data = self.getNonMembers(currentClusterName)
                    # Apply other specified reducing constraints
                    data = self.reduceDF(data, additionalCriteria, useStandards)
                    # Plot depending on how the values should be colored
                    if colorType == None:
                        # Extract desired quantities from data
                        xData = data[xQuantityName].values
                        yData = data[yQuantityName].values
                        # Check if either axis is measuring effective radius for the purpose of unit conversion.
                        if xQuantityName == 're':
                            xData = self.reConvert(data)
                        if yQuantityName == 're':
                            yData = self.reConvert(data)
                        # Check if either axis needs to be put in log scale
                        if useLog[0] == True:
                            xData = np.log10(xData)
                        if useLog[1] == True:
                            yData = np.log10(yData)
                        # Generate the plot on the subplot
                        axes[i][j].scatter(xData, yData, c=color1)
                        # Add fit line
                        if fitLine == True:
                            self.MSRfit(data, useLog, axes, i, j)
                        # Plot van der Wel et al. 2014 line in the case where we are plotting MSR
                        if xQuantityName == 'Mstellar' and yQuantityName == 're':
                            self.plotVanDerWelLines(axes, i, j)
                    elif colorType == 'membership':
                        # Extract desired quantities from data
                        specZ = data[~data['zspec'].isna()]
                        # Assume photZ are those that do not have a specZ
                        photZ = data[~data['cPHOTID'].isin(specZ['cPHOTID'])]
                        specXData = specZ[xQuantityName].values
                        specYData = specZ[yQuantityName].values
                        photXData = photZ[xQuantityName].values
                        photYData = photZ[yQuantityName].values
                        # Check if either axis is measuring effective radius for the purpose of unit conversion.
                        if xQuantityName == 're':
                            specXData = self.reConvert(specZ)
                            photXData = self.reConvert(photZ)
                        if yQuantityName == 're':
                            specYData = self.reConvert(specZ)
                            photYData = self.reConvert(photZ)
                        # Check if either axis needs to be put in log scale
                        if useLog[0] == True:
                            specXData = np.log10(specXData)
                            photXData = np.log10(photXData)
                        if useLog[1] == True:
                            specYData = np.log10(specYData)
                            photYData = np.log10(photYData)
                        # Generate the plot on the subplot
                        axes[i][j].scatter(specXData, specYData, color=color1, label='Spectroscopic z')
                        axes[i][j].scatter(photXData, photYData, color=color2, label='Photometric z')
                        #Add fit line
                        if fitLine == True:
                            self.MSRfit(data, useLog, axes, i, j)
                        # Plot van der Wel et al. 2014 line in the case where we are plotting MSR
                        if xQuantityName == 'Mstellar' and yQuantityName == 're':
                            self.plotVanDerWelLines(axes, i, j)
                    elif colorType == 'passive':
                        # Build passive query string (from van der Burg et al. 2020), limiting mass to > 10^9.7
                        passiveQuery = '(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ) and Mstellar > 5011870000'
                        # Build active query string, limiting mass to > 10^9.5
                        starFormingQuery = '(UMINV <= 1.3) and (VMINJ <= 1.6) and (UMINV <= 0.60+VMINJ) and Mstellar > 3162280000'
                        # Extract desired quantities from data
                        passive = data.query(passiveQuery)
                        starForming = data.query(starFormingQuery)
                        passiveX = passive[xQuantityName].values
                        passiveY = passive[yQuantityName].values
                        starFormingX = starForming[xQuantityName].values
                        starFormingY = starForming[yQuantityName].values
                        # Check if either axis is measuring effective radius for the purpose of unit conversion.
                        if xQuantityName == 're':
                            passiveX = self.reConvert(passive)
                            starFormingX = self.reConvert(starForming)
                        if yQuantityName == 're':
                            passiveY = self.reConvert(passive)
                            starFormingY = self.reConvert(starForming)
                        # Check if either axis needs to be put in log scale
                        if useLog[0] == True:
                            passiveX = np.log10(passiveX)
                            starFormingX = np.log10(starFormingX)
                        if useLog[1] == True:
                            passiveY = np.log10(passiveY)
                            starFormingY = np.log10(starFormingY)
                        # Generate the plot on the subplot
                        axes[i][j].scatter(passiveX, passiveY, color=color1, label='Quiescent')
                        axes[i][j].scatter(starFormingX, starFormingY, color=color2, label='Star Forming')
                        # Add fit line
                        if fitLine == True:
                            self.MSRfit(passive, useLog, axes, i, j, color=color1)
                            self.MSRfit(starForming, useLog, axes, i, j, color=color2)
                        # Plot passive v star-forming border in the case where we are plotting UVJ color-color
                        if xQuantityName == 'VMINJ' and yQuantityName == 'UMINV':
                            self.plotPassiveLines(axes, i, j)
                        # Plot van der Wel et al. 2014 line in the case where we are plotting MSR
                        if xQuantityName == 'Mstellar' and yQuantityName == 're':
                            self.plotVanDerWelLines(axes, i, j)
                    elif colorType == 'sersic':
                        elliptical = data.query('2.5 < n < 6')
                        spiral = data.query('n < 2.5')
                        ellipticalX = elliptical[xQuantityName].values
                        ellipticalY = elliptical[yQuantityName].values
                        spiralX = spiral[xQuantityName].values
                        spiralY = spiral[yQuantityName].values
                        # Check if either axis is measuring effective radius for the purpose of unit conversion.
                        if xQuantityName == 're':
                            ellipticalX = self.reConvert(elliptical)
                            spiralX = self.reConvert(spiral)
                        if yQuantityName == 're':
                            ellipticalY = self.reConvert(elliptical)
                            spiralY = self.reConvert(spiral)
                        # Check if either axis needs to be put in log scale
                        if useLog[0] == True:
                            ellipticalX = np.log10(ellipticalX)
                            spiralX = np.log10(spiralX)
                        if useLog[1] == True:
                            ellipticalY = np.log10(ellipticalY)
                            spiralY = np.log10(spiralY)
                        # Generate the plot
                        axes[i][j].scatter(ellipticalX, ellipticalY, color=color1, label='2.5 < n < 6')
                        axes[i][j].scatter(spiralX, spiralY, color=color2, label='n < 2.5')
                        # generate best fit line
                        if fitLine == True:
                            self.MSRfit(elliptical, useLog, axes, i, j, color=color1)
                            self.MSRfit(spiral, useLog, axes, i, j, color=color2)
                        # Plot van der Wel et al. 2014 line in the case where we are plotting MSR
                        if xQuantityName == 'Mstellar' and yQuantityName == 're':
                            self.plotVanDerWelLines(axes, i, j)
                    else:
                        print(colorType, ' is not a valid coloring scheme!')

                    # Plot configurations for plotType 2
                    axes[i][j].set(xlabel=xLabel, ylabel=yLabel)
                    if (xRange != None):
                        axes[i][j].set(xlim=xRange)
                    if (yRange != None):
                        axes[i][j].set(ylim=yRange)
                    axes[i][j].set(title=currentClusterName)
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
            # Loop over every cluster
            for clusterName in self._structClusterNames:
                # Get all galaxies associated with this cluster
                data = self.getClusterGalaxies(clusterName)
                if useMembers == 'only':
                    # Reduce data to only contain galaxies classified as members
                    data = self.getMembers(clusterName)
                if useMembers == 'not':
                    # Reduce data to only contain galaxies not classified as members
                    data = self.getNonMembers(clusterName)
                # Apply other specified reducing constraints
                data = self.reduceDF(data, additionalCriteria, useStandards)
                 # Plot depending on how the values should be colored
                if colorType == None:
                    # Extract desired quantities from data
                    xData = data[xQuantityName].values
                    yData = data[yQuantityName].values
                    # Check if either axis is measuring effective radius for the purpose of unit conversion.
                    if xQuantityName == 're':
                        xData = self.reConvert(data)
                    if yQuantityName == 're':
                        yData = self.reConvert(data)
                    # Check if either axis needs to be put in log scale
                    if useLog[0] == True:
                        xData = np.log10(xData)
                    if useLog[1] == True:
                        yData = np.log10(yData)
                    # Generate the plot
                    plt.scatter(xData, yData, c=color1)
                elif colorType == 'membership':
                    specZ = data[~data['zspec'].isna()]
                    # Assume photZ are those that do not have a specZ
                    photZ = data[~data['cPHOTID'].isin(specZ['cPHOTID'])]
                    specXData = specZ[xQuantityName].values
                    specYData = specZ[yQuantityName].values
                    photXData = photZ[xQuantityName].values
                    photYData = photZ[yQuantityName].values
                    # Check if either axis is measuring effective radius for the purpose of unit conversion.
                    if xQuantityName == 're':
                        specXData = self.reConvert(specZ)
                        photXData = self.reConvert(photZ)
                    if yQuantityName == 're':
                        specYData = self.reConvert(specZ)
                        photYData = self.reConvert(photZ)
                    # Check if either axis needs to be put in log scale
                    if useLog[0] == True:
                        specXData = np.log10(specXData)
                        photXData = np.log10(photXData)
                    if useLog[1] == True:
                        specYData = np.log10(specYData)
                        photYData = np.log10(photYData)
                    if (clusterName != self._structClusterNames[-1]):
                        plt.scatter(specXData, specYData, color=color1)
                        plt.scatter(photXData, photYData, color=color2)
                    # Only add legend labels for the last plot otherwise the lengend will be filled with multiple duplicates of these labels
                    else:
                        plt.scatter(specXData, specYData, color=color1, label='Spectroscopic z')
                        plt.scatter(photXData, photYData, color=color2, label='Photometric z')
                elif colorType == 'passive':
                    # Build passive query string (from van der Burg et al. 2020), limiting mass to > 10^9.7
                    passiveQuery = '(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ) and Mstellar > 5011870000'
                    # Build active query string, limiting mass to > 10^9.5
                    starFormingQuery = '(UMINV <= 1.3) and (VMINJ <= 1.6) and (UMINV <= 0.60+VMINJ) and Mstellar > 3162280000'
                    # Extract desired quantities from data
                    passive = data.query(passiveQuery)
                    starForming = data.query(starFormingQuery)
                    passiveX = passive[xQuantityName].values
                    passiveY = passive[yQuantityName].values
                    starFormingX = starForming[xQuantityName].values
                    starFormingY = starForming[yQuantityName].values
                    # Check if either axis is measuring effective radius for the purpose of unit conversion
                    if xQuantityName == 're':
                        passiveX = self.reConvert(passive)
                        starFormingX = self.reConvert(starForming)
                    if yQuantityName == 're':
                        passiveY = self.reConvert(passive)
                        starFormingY = self.reConvert(starForming)
                    # Check if either axis needs to be put in log scale
                    if useLog[0] == True:
                        passiveX = np.log10(passiveX)
                        starFormingX = np.log10(starFormingX)
                    if useLog[1] == True:
                        passiveY = np.log10(passiveY)
                        starFormingY = np.log10(starFormingY)
                    # Generate the plot
                    if (clusterName != self._structClusterNames[-1]):
                        plt.scatter(passiveX, passiveY, color=color1)
                        plt.scatter(starFormingX, starFormingY, color=color2)
                    # Only add legend labels for the last plot otherwise the lengend will be filled with multiple duplicates of these labels
                    else:
                        plt.scatter(passiveX, passiveY, color=color1, label='Quiescent')
                        plt.scatter(starFormingX, starFormingY,color=color2, label='Star Forming')
                        # Plot passive v star-forming border in the case where we are plotting UVJ color-color
                        if xQuantityName == 'VMINJ' and yQuantityName == 'UMINV':
                            self.plotPassiveLines()
                elif colorType == 'sersic':
                    elliptical = data.query('2.5 < n < 6')
                    spiral = data.query('n < 2.5')
                    ellipticalX = elliptical[xQuantityName].values
                    ellipticalY = elliptical[yQuantityName].values
                    spiralX = spiral[xQuantityName].values
                    spiralY = spiral[yQuantityName].values
                    # Check if either axis is measuring effective radius for the purpose of unit conversion.
                    if xQuantityName == 're':
                        ellipticalX = self.reConvert(elliptical)
                        spiralX = self.reConvert(spiral)
                    if yQuantityName == 're':
                        ellipticalY = self.reConvert(elliptical)
                        spiralY = self.reConvert(spiral)
                    # Check if either axis needs to be put in log scale
                    if useLog[0] == True:
                        ellipticalX = np.log10(ellipticalX)
                        spiralX = np.log10(spiralX)
                    if useLog[1] == True:
                        ellipticalY = np.log10(ellipticalY)
                        spiralY = np.log10(spiralY)
                    # Generate the plot
                    if (clusterName != self._structClusterNames[-1]):
                        plt.scatter(ellipticalX, ellipticalY, color=color1)
                        plt.scatter(spiralX, spiralY, color=color2)
                    # Only add legend labels for the last plot otherwise the legend will be filled with multiple duplicates of these labels
                    else:
                        plt.scatter(ellipticalX, ellipticalY, color=color1, label='2.5 < n < 6')
                        plt.scatter(spiralX, spiralY, color=color2, label='n < 2.5')
                else:
                    print(colorType, ' is not a valid coloring scheme!')
                    # Return since in the case of plot type 3 it is possible for the remainder of the code to execute otherwise
                    return
            # generate best fit line
            if fitLine == True:
                # In the case of plotting passive vs star forming galaxies, we plot two separate fit lines
                if colorType == 'passive':
                    self.MSRfit([], useLog, allData=True, useMembers=useMembers, additionalCriteria=additionalCriteria, useStandards=useStandards, typeRestrict='passive', color=color1)
                    self.MSRfit([], useLog, allData=True, useMembers=useMembers, additionalCriteria=additionalCriteria, useStandards=useStandards, typeRestrict='starForming', color=color2)
                # In the case of plotting elliptical vs spiral inclined galaxies (based on Sersic index), we plot two separate fit lines NOTE: Handling of these cases not yet implemented in MSRfit()
                elif colorType == 'sersic':
                    self.MSRfit([], useLog, allData=True, useMembers=useMembers, additionalCriteria=additionalCriteria, useStandards=useStandards, typeRestrict='elliptical', color=color1)
                    self.MSRfit([], useLog, allData=True, useMembers=useMembers, additionalCriteria=additionalCriteria, useStandards=useStandards, typeRestrict='spiral', color=color2)
                else:
                    self.MSRfit([], useLog, allData=True, useMembers=useMembers, additionalCriteria=additionalCriteria, useStandards=useStandards)
            # Plot van der Wel et al. 2014 line in the case where we are plotting MSR
            if xQuantityName == 'Mstellar' and yQuantityName == 're':
                self.plotVanDerWelLines()
        else:
            print(plotType, " is not a valid plotting scheme!")

        # Plot configurations for plotType 1 and 3
        # (plotType 2 handles plot configurations for each individual subplot)
        if plotType != 2:
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            if (xRange != None):
                plt.xlim(xRange[0], xRange[1])
            if (yRange != None):
                plt.ylim(yRange[0], yRange[1])
            plt.legend()
            plt.show()
    # END PLOT