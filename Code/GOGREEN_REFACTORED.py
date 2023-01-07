from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rng
import os
import warnings
import scipy.optimize as opt
import scipy.interpolate as interp



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
        allClusterGalaxies = self.getClusterGalaxies(clusterName)
        memberGalaxies = self.getMembers(clusterName)
        nonMemberGalaxies = allClusterGalaxies[~allClusterGalaxies['cPHOTID'].isin(memberGalaxies['cPHOTID'])]
        return nonMemberGalaxies
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
        A = pow(10, 0.7)
        alpha = 0.22
        xVals = np.array([9.5, 11.5])
        MstellarRange = pow(10, xVals)
        yVals = np.log10(np.array([A * pow((i / (5 * np.float_power(10, 10))), alpha) for i in MstellarRange]))
        # In case of subplots, plot for the specific row and column
        if row != None and col != None:
            if axes[row][col] != None:
                axes[row][col].plot(xVals, yVals, linestyle='dashed', color='black')
                return
        # Else plot normally
        plt.plot(xVals, yVals, linestyle='dashed', color='black')
    #END PLOTVANDERWELLINES

    def reConvert(self, data:pd.DataFrame) -> tuple[list, list]:
        """
        reConvert convert effective radius values from units of arcsec to kpc.

        :param data:   The set of data being used by the calling function, plot().
        :return   :    returns the list of converted effective radius values

        """
        if data['re'].values.shape == (0,):
            # If there are no values, return empty array so attempting to convert does not cause a crash
            return [], []
        sizes = data['re'].values * (cosmo.kpc_proper_per_arcmin(data['zspec'].values)/60) #converting all effective radii from units of arcsec to kpc using their spectroscopic redshifts
        sigmas = data['re_err'].values * (cosmo.kpc_proper_per_arcmin(data['zspec'].values)/60)
        for i in range(0, len(sizes)):
            if np.isnan(sizes[i]): #checking where conversion failed due to lack of zspec value
                sizes[i] = data['re'].values[i] * (cosmo.kpc_proper_per_arcmin(data['zphot'].values[i])/60) #use photometric redshifts instead where there are no spectroscopic redshifts
        for i in range(0, len(sigmas)):
            if np.isnan(sigmas[i]): #checking where conversion failed due to lack of zspec value
               sigmas[i] = data['re_err'].values[i] * (cosmo.kpc_proper_per_arcmin(data['zphot'].values[i])/60) #use photometric redshifts instead where there are no spectroscopic redshifts
        sizes = (sizes / u.kpc) * u.arcmin # removing units so the data can be used in the functions below
        sigmas = (sigmas / u.kpc) * u.arcmin # removing units so the data can be used in the functions below
        return sizes, sigmas
    #END RECONVERT

    def MSRfit(self, data:list, useLog:list=[False, False], axes:list=None, row:int=None, col:int=None, allData:bool=False, useMembers:str='only', additionalCriteria:list=None, useStandards:bool=True, typeRestrict:str=None, color:str='black'):
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
                                     Value:   'Quiescent' - only passive galaxies should be considered.
                                     Value:   'Star-Forming' - only star forming galaxies should be considered.
                                     Value:   'Elliptical' - only galaxies with 2.5 < n < 6 should be considered.
                                     Value:   'Spiral' - only galaxies with n < 2.5 should be considered.
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
            # Establish colors
            if typeRestrict == 'Quiescent':
                data = data.query('(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)')
                color = 'red'
            # Handling case where only star forming galaxies out of all data need to plotted.
            elif typeRestrict == 'Star-Forming':
                data = data.query('(UMINV <= 1.3) or (VMINJ >= 1.6) or (UMINV <= 0.60+VMINJ)')
                color = 'blue'
            elif typeRestrict == 'Elliptical':
                data = data.query('2.5 < n < 6')
                color = 'green'
            # Handling case where only star forming galaxies out of all data need to plotted.
            elif typeRestrict == 'Spiral':
                data = data.query('n < 2.5')
                color = 'orange'
        # Establish label
        if typeRestrict == None:
            lbl = "stellar mass-size relation trend"
        else:
            lbl = typeRestrict + " stellar mass-size relation trend"
        # Convert all effective radii and associated errors from units of arcsec to kpc using their spectroscopic redshifts
        size, sigmas = self.reConvert(data)
        # Extract mass values
        mass = data['Mstellar'].values
        # Calculate coefficients (slope and y-intercept)
        xFitData = mass
        yFitData = size
        if useLog[0] == True:
            xFitData = np.log10(xFitData)
        if useLog[1] == True:
            yFitData = np.log10(yFitData)
            upperSigmas = np.log10(size + sigmas) - np.log10(size)
            lowerSigmas = np.log10(size) - np.log10(size - sigmas)
            sigmas = (upperSigmas + lowerSigmas)/2
        # Transform error values into weights
        weights = 1/np.array(sigmas)
        for i in range(0, len(weights)): # Explanation of the error that provoked this check: https://predictdb.org/post/2021/07/23/error-linalgerror-svd-did-not-converge/
            if np.isinf(weights[i]):
                weights[i] = 0 #setting to 0 because this data point should not be used
            if np.isnan(weights[i]):
                weights[i] = 0 #setting to 0 because this data point should not be used
        s = np.polynomial.polynomial.Polynomial.fit(x=xFitData, y=yFitData, deg=1, w=weights) # I have no idea what rules this return value conforms to. the man page for fit() calls it a series. It is callable as a function. The syntax displayed is not familiar to me (x**1).
        #vals, cov = opt.curve_fit(f=(lambda x, m, b: b + m*x), xdata=xFitData, ydata=yFitData, p0=[0, 0], sigma=sigmas)
        if row != None and col != None:
            # Check for subplots
            if axes[row][col] != None:
                # Bootstrapping calculation
                self.bootstrap(xFitData, yFitData, weights, axes, row, col, typeRestrict)
                # Add white backline in case of plotting multiple fit lines in one plot
                if color != 'black':
                    axes[row][col].plot(xFitData, s(xFitData), color='white', linewidth=4)
                # Plot the best fit line
                axes[row][col].plot(xFitData, s(xFitData), color=color, label=lbl)
                return
        # Bootstrapping calculation
        self.bootstrap(xFitData, yFitData, weights, axes, row, col, typeRestrict)
        # Add white backline in case of plotting multiple fit lines in one plot
        if color != 'black':
            plt.plot(xFitData, s(xFitData), color='white', linewidth=4)
        # Plot the best fit line
        plt.plot(xFitData, s(xFitData), color=color, label=lbl)
    # END MSRFIT

    def bootstrap(self, x:list=None, y:list=None, error:list=None, axes:list=None, row:int=None, col:int=None, typeRestrict:str=None):
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
        :param typeRestrict:        Flag to indicate what color should be used, depending on type of data being plotted
                                     Default: None
                                     Value:   'Quiescent' - only passive galaxies are being considered.
                                     Value:   'Star-Forming' - only star forming galaxies are being considered.
                                     Value:   'Elliptical' - only galaxies with 2.5 < n < 6 are being considered.
                                     Value:   'Spiral' - only galaxies with n < 2.5 are being considered.
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
                    #vals, cov = opt.curve_fit(f=(lambda x, m, b: b + m*x), xdata=bootstrapX, ydata=bootstrapY, p0=[0, 0], sigma=boostrapE)
                    s = np.polynomial.polynomial.Polynomial.fit(x=bootstrapX, y=bootstrapY, deg=1, w=boostrapE)
                    coefs = s.convert().coef
                    b = coefs[0]
                    m = coefs[1]
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
        if typeRestrict == 'Quiescent':
            color = [0.5, 0, 0] # darker red
        elif typeRestrict == 'Elliptical':
            color = [0, 0.5, 0] # darker green
        elif typeRestrict == 'Spiral':
            color = [1, 0.5, 0] # lighter orange
        # star-forming and default case
        else:
            color = [0, 0, 0.5] # darker blue
        # Plot curves on top and bottom of intervals
        plot.plot(xGrid, yTops, color=color)
        plot.plot(xGrid, yBots, color=color)
        plot.fill_between(xGrid, yBots, yTops, color=color, alpha=0.5) # https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html
    # END BOOTSTRAP

    def getRatio(self, category:str='SF', x:float=None, y:float=None, plotLines:bool=False, xRange:list=None, yRange:list=None) -> list:
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
            # Separate data sets by SFR
            memberDataQ = memberData.query('(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)')
            memberDataSF = memberData.query('(UMINV <= 1.3) or (VMINJ >= 1.6) or (UMINV <= 0.60+VMINJ)')
            nonMemberDataQ = nonMemberData.query('(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)')
            nonMemberDataSF = nonMemberData.query('(UMINV <= 1.3) or (VMINJ >= 1.6) or (UMINV <= 0.60+VMINJ)')
            # Convert all effective radii from units of arcsec to kpc using their spectroscopic redshifts
            memberSizeQ = self.reConvert(memberDataQ)
            memberSizeSF = self.reConvert(memberDataSF)
            nonMemberSizeQ = self.reConvert(nonMemberDataQ)
            nonMemberSizeSF = self.reConvert(nonMemberDataSF)
            # Extract all mass values
            memberMassQ = memberDataQ['Mstellar'].values
            memberMassSF = memberDataSF['Mstellar'].values
            nonMemberMassQ = nonMemberDataQ['Mstellar'].values
            nonMemberMassSF = nonMemberDataSF['Mstellar'].values
            # Cut out data points that will cause an error
            memberMassQ, memberSizeQ = self.cutBadData(memberMassQ, memberSizeQ)
            memberMassSF, memberSizeSF = self.cutBadData(memberMassSF, memberSizeSF)
            nonMemberMassQ, nonMemberSizeQ = self.cutBadData(nonMemberMassQ, nonMemberSizeQ)
            nonMemberMassSF, nonMemberSizeSF = self.cutBadData(nonMemberMassSF, nonMemberSizeSF)
            # Convert to log
            memberMassQ = np.log10(memberMassQ)
            memberSizeQ = np.log10(memberSizeQ)
            memberMassSF = np.log10(memberMassSF)
            memberSizeSF = np.log10(memberSizeSF)
            nonMemberMassQ = np.log10(nonMemberMassQ)
            nonMemberSizeQ = np.log10(nonMemberSizeQ)
            nonMemberMassSF = np.log10(nonMemberMassSF)
            nonMemberSizeSF = np.log10(nonMemberSizeSF)
            # Calculate slope and intercept for all 4 data sets
            mMemberQ, bMemberQ = np.polyfit(memberMassQ, memberSizeQ, 1)
            mMemberSF, bMemberSF = np.polyfit(memberMassSF, memberSizeSF, 1)
            mNonMemberQ, bNonMemberQ = np.polyfit(nonMemberMassQ, nonMemberSizeQ, 1)
            mNonMemberSF, bNonMemberSF = np.polyfit(nonMemberMassSF, nonMemberSizeSF, 1)
            # Plot trendlines together
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
        else:
            print(category + " is not a valid category")
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
                        xCountMSR, yCountMSR = self.plot('Mstellar', 're', plotType=p, clusterName=cluster, useMembers=m, colorType=c, useLog=[True,True], xRange = [9.5, 11.5], yRange = [-1.5, 1.5], xLabel='log(Mstellar)', yLabel='log(Re)', fitLine=False)
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
    
    def plotUnwrapped(self, xQuantityName:str, yQuantityName:str, additionalCriteria:list=None, colorType:str=None, useStandards:bool=True, useLog:list=[False,False], fitLine:bool=False, 
        data:pd.DataFrame=None, color1:list=None, color2:list=None, plot=None, axes:list=None, row:int=None, col:int=None, holdLbls:bool=False):
            """
            Helper function called by plot. Handles the plotting of data.
                
            :param xQuantityName:      Name of the column whose values are to be used as the x
            :param yQuantityName:      Name of the column whose values are to be used as the y
            :param additionalCriteria: List of desired criteria the plotted galaxies should meet
                                        Default: None
            :param colorType:          Specifies how to color code the plotted galaxies
                                        Default: None
                                        Value:   'membership' - spectroscopic member vs photometric member
                                        Value:   'passive' - passive vs star forming
            :param useStandards:       Flag to indicate whether the standard search criteria should be applied
                                        Default: True
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
            aData = data
            bData = data
            aLbl = None
            bLbl = None
            # Overwrite variables according to coloring scheme
            if colorType == None:
                # Don't need to do anything for this case. Included so program proceeds as normal
                pass
            elif colorType == 'membership':
                # Extract desired quantities from data
                aData = data[~data['zspec'].isna()]
                aLbl = 'Spectroscopic z'
                # Assume photZ are those that do not have a specZ
                bData = data[~data['cPHOTID'].isin(aData['cPHOTID'])]
                bLbl = 'Photometric z'
            elif colorType == 'passive':
                # Build passive query string (from van der Burg et al. 2020)
                passiveQuery = '(UMINV > 1.3) and (VMINJ < 1.6) and (UMINV > 0.60+VMINJ)'
                # Build active query string
                starFormingQuery = '(UMINV <= 1.3) or (VMINJ >= 1.6) or (UMINV <= 0.60+VMINJ)'
                # Extract desired quantities from data
                passive = data.query(passiveQuery)
                starForming = data.query(starFormingQuery)
                # Need to reduce again, as for some reason query is pulling from the unedited data despite us having reduced previously. 
                aData = self.reduceDF(passive, additionalCriteria, useStandards)
                aLbl = 'Quiescent'
                bData = self.reduceDF(starForming, additionalCriteria, useStandards)
                bLbl = 'Star Forming'
            elif colorType == 'GV':
                # Build gv query string (from McNab et al 2021)
                gvQuery = '(2 * VMINJ + 1.1 <= nuv_tot - V_tot) and (nuv_tot - V_tot <= 2 * VMINJ + 1.6)' # 2(𝑉 − 𝐽) + 1.1 ≤ (𝑁𝑈𝑉 − 𝑉 ) ≤ 2(𝑉 − 𝐽) + 1.6
                # Build non-gv query string
                otherQuery = '(2 * VMINJ + 1.1 > nuv_tot - V_tot) or (nuv_tot - V_tot > 2 * VMINJ + 1.6)'
                # Extract desired quantities from data
                greenValley = data.query(gvQuery)
                other = data.query(otherQuery)
                # Need to reduce again, as for some reason query is pulling from the unedited data despite us having reduced previously. 
                aData = self.reduceDF(greenValley, additionalCriteria, useStandards)
                aLbl = 'Green Valley'
                bData = self.reduceDF(other, additionalCriteria, useStandards)
                bLbl = 'Other'
            elif colorType == 'BQ':
                # Build bq query string (from McNab et al 2021)
                bqQuery = '((VMINJ + 0.45 <= UMINV) and (UMINV <= VMINJ + 1.35)) or ((-1.25 * VMINJ + 2.025 <= UMINV) and (UMINV <= -1.25 * VMINJ + 2.7))' # (𝑉 − 𝐽) + 0.45 ≤ (𝑈 − 𝑉 ) ≤ (𝑉 − 𝐽) + 1.35 ### − 1.25 (𝑉 − 𝐽) + 2.025 ≤ (𝑈 − 𝑉 ) ≤ −1.25 (𝑉 − 𝐽) + 2.7 
                # Build non-bq query string
                otherQuery = '((VMINJ + 0.45 > UMINV) or (UMINV > VMINJ + 1.35)) and ((-1.25 * VMINJ + 2.025 > UMINV) or (UMINV > -1.25 * VMINJ + 2.7))'
                # Extract desired quantities from data
                blueQuiescent = data.query(bqQuery)
                other = data.query(otherQuery)
                # Need to reduce again, as for some reason query is pulling from the unedited data despite us having reduced previously. 
                aData = self.reduceDF(blueQuiescent, additionalCriteria, useStandards)
                aLbl = 'Blue Quiescent'
                bData = self.reduceDF(other, additionalCriteria, useStandards)
                bLbl = 'Other'
            elif colorType == 'PSB': # we need to query the redshift catalogue instead of the structural catalogue, as this is the catalogue with d4000 and delta_BIC values
                # Build gv query string (from McNab et al 2021)
                gvQuery = 'd4000 < 1.45 and delta_BIC < -10' # (D4000 < 1.45) ∩ (ΔBIC < −10) 
                # Build non-gv query string
                otherQuery = 'd4000 >= 1.45 or delta_BIC >= -10'
                # Extract desired quantities from data
                greenValley = data.query(gvQuery)
                other = data.query(otherQuery)
                # Need to reduce again, as for some reason query is pulling from the unedited data despite us having reduced previously. 
                aData = self.reduceDF(greenValley, additionalCriteria, useStandards)
                aLbl = 'Post-starburst'
                bData = self.reduceDF(other, additionalCriteria, useStandards)
                bLbl = 'Other' 
            elif colorType == 'sersic':
                # Build elliptical query string
                elliptical = data.query('2.5 < n < 6')
                # Build spiral query string
                spiral = data.query('n <= 2.5')
                # Unsure if need to reduce again. Doing it to be on the safe side.
                aData = self.reduceDF(elliptical, additionalCriteria, useStandards)
                aLbl = 'Elliptical'
                bData = self.reduceDF(spiral, additionalCriteria, useStandards)
                bLbl = 'Spiral'
            else:
                print(colorType, ' is not a valid coloring scheme!')
                return
            # Overwrite labels as needed
            if holdLbls:
                aLbl = None
                bLbl = None
            # Check if either axis is measuring effective radius for the purpose of unit conversion, if not assign values directly
            if xQuantityName == 're':
                aXVals, _ = self.reConvert(aData)
                bXVals, _ = self.reConvert(bData)
            else:
                aXVals = aData[xQuantityName].values
                bXVals = bData[xQuantityName].values
            if yQuantityName == 're':
                aYVals, _ = self.reConvert(aData)
                bYVals, _ = self.reConvert(bData)
            else:
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
            if fitLine == True:
                # Generate two if plotting quiescent v star-forming
                if colorType == 'passive':
                    self.MSRfit(aData, useLog, axes, row, col, color=color1)
                    self.MSRfit(bData, useLog, axes, row, col, color=color2)
                else:
                    self.MSRfit(data, useLog, axes, row, col)
            # Generate the plot
            plot.scatter(aXVals, aYVals, alpha=0.5, color=color1, label=aLbl)
            if colorType != None:
                plot.scatter(bXVals, bYVals, alpha=0.5, color=color2, label=bLbl)
            # Plot van der Wel et al. 2014 line in the case where we are plotting MSR
            #if xQuantityName == 'Mstellar' and yQuantityName == 're':
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
        useStandards:bool=True, xRange:list=None, yRange:list=None, xLabel:str='', yLabel:str='', useLog:list=[False,False], fitLine:bool=False):
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
        # Check if plot colors were provided by the user
        if colors != None:
            color1 = colors[0]
            color2 = colors[1]
        # If not, generate random colors
        else:
            if colorType == 'passive':
                color1 = [1, 0, 0]
                color2 = [0, 0, 1]
            else:
                color1 = [0, 1, 0.5]
                color2 = [1, 0.5, 0]
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
            # Apply other specified reducing constraints
            data = self.reduceDF(data, additionalCriteria, useStandards)
            # Plot data
            xTot, yTot = self.plotUnwrapped(xQuantityName, yQuantityName, additionalCriteria, colorType, useStandards, useLog, fitLine, data, color1, color2, plt)
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
                    # Apply other specified reducing constraints
                    data = self.reduceDF(data, additionalCriteria, useStandards)
                    # Plot data
                    x, y = self.plotUnwrapped(xQuantityName, yQuantityName, additionalCriteria, colorType, useStandards, useLog, fitLine, data, color1, color2, axes[i][j], axes, i, j)
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
            # Initialize data count totals (used when running test suite)
            xTot = 0
            yTot = 0
            # Loop over every cluster
            for clusterName in self._structClusterNames:
                # Get all galaxies associated with this cluster
                data = self.getClusterGalaxies(clusterName)
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
                # Apply other specified reducing constraints
                data = self.reduceDF(data, additionalCriteria, useStandards)
                # Plot depending on how the values should be colored (hold off on MSR fit lines since this needs to be handled separately for plotType 3)
                if (clusterName != self._structClusterNames[-1]):
                    x, y = self.plotUnwrapped(xQuantityName, yQuantityName, additionalCriteria, colorType, useStandards, useLog, False, data, color1, color2, plt, axes=None, row=None, col=None, holdLbls=True)
                # Only add legend labels for the last plot otherwise the lengend will be filled with multiple duplicates of these labels
                else:
                    x, y = self.plotUnwrapped(xQuantityName, yQuantityName, additionalCriteria, colorType, useStandards, useLog, False, data, color1, color2, plt, axes=None, row=None, col=None, holdLbls=False)
                # Update data count totals
                xTot+=x
                yTot+=y
            # generate best fit line
            if fitLine == True:
                # In the case of plotting passive vs star forming galaxies, we plot two separate fit lines
                if colorType == 'passive':
                    self.MSRfit([], useLog, allData=True, useMembers=useMembers, additionalCriteria=additionalCriteria, useStandards=useStandards, typeRestrict='Quiescent', color=color1)
                    self.MSRfit([], useLog, allData=True, useMembers=useMembers, additionalCriteria=additionalCriteria, useStandards=useStandards, typeRestrict='Star-Forming', color=color2)
                # In the case of plotting elliptical vs spiral inclined galaxies (based on Sersic index), we plot two separate fit lines NOTE: Handling of these cases not yet implemented in MSRfit()
                elif colorType == 'sersic':
                    self.MSRfit([], useLog, allData=True, useMembers=useMembers, additionalCriteria=additionalCriteria, useStandards=useStandards, typeRestrict='Elliptical', color=color1)
                    self.MSRfit([], useLog, allData=True, useMembers=useMembers, additionalCriteria=additionalCriteria, useStandards=useStandards, typeRestrict='Spiral', color=color2)
                else:
                    self.MSRfit([], useLog, allData=True, useMembers=useMembers, additionalCriteria=additionalCriteria, useStandards=useStandards)
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