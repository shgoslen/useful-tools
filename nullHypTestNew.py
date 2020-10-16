# nullHypTest
# Author Steve Goslen
# June 2018 
# This is a re-write of the original null Hype test program
# It takes 3 arguments 1: Baseline software version
#                      2: Target software version
#                      3: type of report to produce
# The report type can be "CV", "FULL", "endReport", or "ALL"
# "CV" generates an xl spreadsheet that shows the tests with high CV values that needs scrubbing
# "CSV" generates an CSV file that shows only the failed results for importing into the test storage tool
# "FULL" generates an LARGE xl spreadsheet that shows the full set of data
# "endReport" generates an xl spreadsheet that is to be sent external to the performance team for reporting
# "Marketing" generates an xl spreadsheet that contains information for marketing teams
# "Exec" generates an xl spreadsheet with a smaller number of test cases
# "ALL" generates all reports, and is mainly for debugging purposes
# 
# We use a 2 sample null hypothesis test along with calculating the effect size in order
# to determine if there is a degradation which will be marked as FAILED. All other
# cases are marked as PASSED


# Import what we need to get data and crunch numbers
import re
import math
from pymongo import MongoClient
from bson.json_util import dumps
import sys
import datetime
from datetime import date
import collections as col
import statistics

import numpy as np
import pandas as pd
import scipy.stats

# Get the arguments passed into the program

baselineSWV = sys.argv[1]
targetSWV = sys.argv[2]
platFam = sys.argv[3]
testType = sys.argv[4]

# Need to setup a log file
logFilePath = '/auto/tools/dataAnalysis/logs/daLog.txt'
reportsPath = '/auto/tools/dataAnalysis/reports/'
logFile = open(logFilePath, 'a')

now = date.today()
timeStamp = now.strftime("%Y-%m-%d %H:%M:%S")
logFile.write("Start: " + timeStamp + "********************************\n")
logFile.write("baseLine: " + str(baselineSWV) + "\n")
logFile.write("target: " + str(targetSWV) + "\n")
logFile.write("platFam: " + str(platFam) + "\n")
logFile.write("testType: " + str(testType) + "\n")

# These values are used for power calculations and for the null hypotheses as well
# Our Null Hypotheses is that the performance of the target software version
# is the same as the baseline software version
alpha = 0.05
beta = 1 - alpha

# Below is a smaller list of test cases that higher level managers seem to be interested in
# The idea is to present a smaller list of tests as opposed to the "weekly" or "full"
# set in order to make it easier for folks to digest the performance reports
basicTestCases = ['test01', 'test01a', 'test01v6', 'test03', 'test06ad1', 'test09bn', 'test09n', 'test11n', 'test14a', 'test15a', 'test24',
                  'test68-om', 'test85i']
basicTestCasesR1K = ['test01wl', 'test01awl', 'test01v6wl', 'test03wl', 'test06ad1wl', 'test09bnwl', 'test09nwl', 'test11nwl',
                       'test14awl', 'test15awl', 'test24wl', 'test68-omwl']


##############################################################################################
def doPower(mean, sigma, alpha, beta):
    # This function will return the number samples that we need to have of the target software
    # image in order to have enough for the results to be valid
    # print("mean")
    # print(mean)
    # print("sigma")
    # print(sigma)
    # print("alpha")
    # print(alpha)
    # print("beta")
    # print(beta)

    # If sigma is NaN return a sample size of 2
    if math.isnan(sigma):
        return 2

    mean2 = mean * 0.95
    group_means = [mean, mean2]
    group_sigmas = [sigma, sigma]

    n_groups = len(group_means)

    # number of simulations
    n_sims = 200

    for samples in range(2, 40):
        # store the p value for each simulation
        sim_p = np.empty(n_sims)
        sim_p.fill(np.nan)

        for i_sim in range(n_sims):

            data = np.empty([samples, n_groups])
            data.fill(np.nan)

            # simulate the data for this 'experiment'
            for i_group in range(n_groups):
                data[:, i_group] = np.random.normal(
                    loc=group_means[i_group],
                    scale=group_sigmas[i_group],
                    size=samples
                )

            # result = scipy.stats.ttest_ind(data[:, 0], data[:, 1])
            result = scipy.stats.ttest_1samp(data[:, 1], mean)
            # Should be using ttest_1sampl test for this power calcuation since
            # our "keystone" data is mean0 or mean naught.kkkkkk

            # Since we only need to look at the lower tail, divide "p" by 2
            sim_p[i_sim] = result[1] / 2

        # number of simulations where the null was rejected
        n_rej = np.sum(sim_p < alpha)

        prop_rej = n_rej / float(n_sims)
        # print("Power: ", prop_rej)
        if prop_rej > (beta):
            break

    # print("Final Power: ", prop_rej, "Samples: ", samples)
    return samples


##############################################################################################
def doPower6Sigma(mean, sigma, alpha, beta):
    # This function will return the number samples that we need to have of the target software
    # image in order to have enough for the results to be valid
    # print("mean")
    # print(mean)
    # print("sigma")
    # print(sigma)
    # print("alpha")
    # print(alpha)
    # print("beta")
    # print(beta)

    # If sigma is NaN or 0 return a sample size of 3
    if math.isnan(sigma) or (sigma == 0):
        return 3

    # So we want to see if the means are 6 Sigmas apart
    mean2 = mean - 6 * sigma
    group_means = [mean, mean2]
    group_sigmas = [sigma, sigma]

    n_groups = len(group_means)

    # number of simulations
    n_sims = 200

    for samples in range(2, 40):
        # store the p value for each simulation
        sim_p = np.empty(n_sims)
        sim_p.fill(np.nan)

        for i_sim in range(n_sims):

            data = np.empty([samples, n_groups])
            data.fill(np.nan)

            # simulate the data for this 'experiment'
            for i_group in range(n_groups):
                data[:, i_group] = np.random.normal(
                    loc=group_means[i_group],
                    scale=group_sigmas[i_group],
                    size=samples
                )

            # result = scipy.stats.ttest_ind(data[:, 0], data[:, 1])
            result = scipy.stats.ttest_1samp(data[:, 1], mean)
            # Should be using ttest_1sampl test for this power calculation since
            # our "keystone" data is mean0 or mean naught

            # Since we only need to look at the lower tail, divide "p" by 2
            sim_p[i_sim] = result[1] / 2

        # number of simulations where the null was rejected
        n_rej = np.sum(sim_p < alpha)

        prop_rej = n_rej / float(n_sims)
        # print("Power: ", prop_rej)
        if prop_rej > (beta):
            break

    # print("Final Power: ", prop_rej, "Samples: ", samples)
    return samples


# Connect to the database
conn = MongoClient('mongodb://perfdb:27017/')
results = conn.get_database('perf').get_collection('results')
testcases = conn.get_database('perf').get_collection('testcases')

# Now find the number of unique test cases
testCases = set()
testCases = testcases.distinct("_id")
logFile.write("Number of testCases:" + str(len(testCases)) + "\n")

# Now find the unique set of platforms that match the target software image
if (platFam == "R4K"):
    plats = "R4"
    if (testType == "Exec"):
        testCases = basicTestCases
if (platFam == "R1K"):
    plats = "C11"
    if (testType == "Exec"):
        testCases = basicTestCasesR1K
if (platFam == "AR"):
    plats = "AR1"
    if (testType == "Exec"):
        testCases = basicTestCases
if (platFam == "RV"):
    plats = "RV"
    if (testType == "Exec"):
        testCases = basicTestCases
if (platFam == "R9K"):
    plats = "R9K1"
if (platFam == "R8K"):
    plats = "R8K3"

platforms = set()
platforms = results.distinct("Platform", {"Platform": {"$regex": plats}, "Software Version": {"$regex": targetSWV}})
logFile.write("Number of platforms:" + str(len(platforms)) + "\n")

# Now for each test case get the throughput metrics for those tests
blnData = []
tarData = []

# Setup a default printing option for data frames
pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Define the data frame that we will use to store all of the calculations
blnResultsData = pd.DataFrame()
tarResultsData = pd.DataFrame()

print("Gathering Data from Mongo")

# Pull out the data that we want from Mongo
for plat in platforms:
    for test in testCases:
        # Get the data for the baseline software release
        blnTestCaseResults = results.find(
            {"Software Version": baselineSWV, "Platform": plat, "Title": test, "Status": "passed"},
            {"_id": 0, "THROUGHPUT_RATE_BITS": 1, "THROUGHPUT_DRIVER_BANDWIDTH": 1})

        # Get the data for the target software release
        tarTestCaseResults = results.find(
            {"Software Version": {"$regex": targetSWV}, "Platform": plat, "Title": test, "Status": "passed"},
            {"_id": 0, "THROUGHPUT_RATE_BITS": 1, "THROUGHPUT_DRIVER_BANDWIDTH": 1})

        # Create a "key" for the dictionary that is the testname combined with the platform name
        blnData = []
        key = str(test) + "_" + str(plat)
        # print("BLN results" + str(len(blnTestCaseResults)))
        for rslt in blnTestCaseResults:
            if "THROUGHPUT_RATE_BITS" in rslt:
                blnData.append(rslt["THROUGHPUT_RATE_BITS"])
            else:
                if "THROUGHPUT_DRIVER_BANDWIDTH" in rslt:
                    blnData.append(rslt["THROUGHPUT_DRIVER_BANDWIDTH"])

        # Now get this into a temp dataframe
        if (blnData):
            # print(blnData)
            blnDataDict = {key: tuple(blnData)}
            # print(blnDataDict)
            tempDf = pd.DataFrame(blnDataDict)
            # print(tempDf)
            # Now add that to the main results dataframe
            blnResultsData = blnResultsData.join(tempDf, how='outer')

        tarData = []
        # print("TAR results" + str(len(tarTestCaseResults)))
        for rslt in tarTestCaseResults:
            if "THROUGHPUT_RATE_BITS" in rslt:
                tarData.append(rslt["THROUGHPUT_RATE_BITS"])
            else:
                if "THROUGHUT_DRIVER_BANDWIDTH" in rslt:
                    tarData.append(rslt["THROUGHPUT_DRIVER_BANDWIDTH"])

        # Now get this into a dataframe
        if (tarData):
            # print(tarData)
            tarDataDict = {key: tuple(tarData)}
            # print(tarDataDict)
            tempDf = pd.DataFrame(tarDataDict)
            # print(tempDf)
            tarResultsData = tarResultsData.join(tempDf, how='outer')

# Now that the raw data is in our data frames print out those data frames
# print("Baseline Results Data Frame")
# print(blnResultsData)
# print("Target Results Data Frame")
# print(tarResultsData)

# blnCols=blnResultsData.columns
# print(blnCols)
# print("Baseline Columns")

# Create a list for all of the columns that we will define for the data Frame
resultsCols = 'Test', 'Platform', 'Bln Mean', 'Bln Num', 'Bln StdDev', 'Required Sample Size', 'Sample Size 2', 'Tar Mean', 'Tar Num', 'Tar StdDev', 'Bln CV', 'Tar CV', 'Bln Max', 'Bln Min', 'Bln 3sig Max', 'Bln 3sig Min', 'Bln Low Bound', 'Bln Upp Bound', 't 1samp', 'p 1samp', 't 2samp', 'p 2samp', 'pct Delta', 'Effect Size', 'Pooled Effect Size', 'Mean Diff', 'Null Hype 1samp', 'Null Hype 2samp', 'Bounds', '2 samp effect'

# Do some calculations over the dataframe, and create new data Frames
print("Process baseline data")
trans = blnResultsData.T
dfBlnMean = pd.DataFrame(trans.mean(axis=1), columns=['Bln Mean'])
# print("dfBlnMean")
# print(dfBlnMean)

dfBlnStdDev = pd.DataFrame(trans.std(axis=1, ddof=0), columns=['Bln Std Dev'])
# print("dfBlnstdDev")
# print(dfBlnStdDev)

dfBlnCount = pd.DataFrame(trans.count(axis=1), columns=['Bln Num'])
# print("dfBlnCount")
# print(dfBlnCount)

dfBlnMax = pd.DataFrame(trans.max(axis=1), columns=['Bln Max'])
# print("dfBlnMax")
# print(dfBlnMax)

dfBlnMin = pd.DataFrame(trans.min(axis=1), columns=['Bln Min'])
# print("dfBlnMin")
# print(dfBlnMin)

dfBln3Sigma = dfBlnStdDev.mul(3)
dfBln3Sigma.columns = ['Bln 3Sigma']
# print("dfBln3Sigma")
# print(dfBln3Sigma)

dfBlnMax3Sigma = dfBlnMean.add(dfBln3Sigma['Bln 3Sigma'], axis=0)
dfBlnMax3Sigma.columns = ['Bln 3Sig Max']
# print("dfBlnMax3Sigma")
# print(dfBlnMax3Sigma)

dfBlnMin3Sigma = dfBlnMean.sub(dfBln3Sigma['Bln 3Sigma'], axis=0)
dfBlnMin3Sigma.columns = ['Bln 3Sig Min']
# print("dfBlnMin3Sigma")
# print(dfBlnMin3Sigma)

dfBlnCV = dfBlnStdDev.div(dfBlnMean['Bln Mean'], axis=0)
dfBlnCV = dfBlnCV.mul(100)
dfBlnCV.columns = ['Bln CV']
# print("dfBlnCV")
# print(dfBlnCV)


# print("Do power calculations")
# Now do power calculations
dfBlnSamples = pd.DataFrame(index=dfBlnMean.index, columns=['Tar Samples'])
for i in dfBlnMean.index:
    # Pull out the mean and standard deviation
    # print("i")
    # print(i)
    samples = doPower(dfBlnMean.loc[i, 'Bln Mean'], dfBlnStdDev.loc[i, 'Bln Std Dev'], alpha, beta)
    dfBlnSamples.loc[i, 'Tar Samples'] = samples
# print("dfBlnSamples")
# print(dfBlnSamples)

# print("Do power calculations based on 6 Sigma")
# Now do power calculations
dfBln6SigmaSamples = pd.DataFrame(index=dfBlnMean.index, columns=['Tar 6Sigma Samp'])
for i in dfBlnMean.index:
    # Pull out the mean and standard deviation
    # print("i")
    # print(i)
    samples = doPower6Sigma(dfBlnMean.loc[i, 'Bln Mean'], dfBlnStdDev.loc[i, 'Bln Std Dev'], alpha, beta)
    dfBln6SigmaSamples.loc[i, 'Tar 6Sigma Samp'] = samples
# print("dfBln6SigmaSamples")
# print(dfBln6SigmaSamples)

print("Process target data")
trans = tarResultsData.T
dfTarMean = pd.DataFrame(trans.mean(axis=1), columns=['Tar Mean'])
# print("dfTarMean")
# print(dfTarMean)

dfTarStdDev = pd.DataFrame(trans.std(axis=1, ddof=0), columns=['Tar Std Dev'])
# print("dfTarStdDev")
# print(dfTarStdDev)

dfTarCount = pd.DataFrame(trans.count(axis=1), columns=['Tar Num'])
# print("dfTarCount")
# print(dfTarCount)

dfTarMax = pd.DataFrame(trans.max(axis=1), columns=['Tar Max'])
# print("dfTarMax")
# print(dfTarMax)

dfTarMin = pd.DataFrame(trans.min(axis=1), columns=['Tar Min'])
# print("dfTarMin")
# print(dfTarMin)

dfTar3Sigma = dfTarStdDev.mul(3)
dfTar3Sigma.columns = ['Tar 3Sigma']
# print("dfTar3Sigma")
# print(dfTar3Sigma)

dfTarMax3Sigma = dfTarMean.add(dfTar3Sigma['Tar 3Sigma'], axis=0)
dfTarMax3Sigma.columns = ['Tar 3Sig Max']
# print("dfTarMax3Sigma")
# print(dfTarMax3Sigma)

dfTarMin3Sigma = dfTarMean.sub(dfTar3Sigma['Tar 3Sigma'], axis=0)
dfTarMin3Sigma.columns = ['Tar 3Sig Min']
# print("dfTarMin3Sigma")
# print(dfTarMin3Sigma)

dfTarCV = dfTarStdDev.div(dfTarMean['Tar Mean'], axis=0)
dfTarCV = dfTarCV.mul(100)
dfTarCV.columns = ['Tar CV']
# print("dfTarCV")
# print(dfTarCV)

# Now build a Full Results data frame to contain everything
# print("Build Full Results Data Frame")
dfFullResults = pd.DataFrame(index=dfBlnMean.index)
# Need to take the index and split that apprt into platform and test so that
# we can add a "Test" column, and "Platform" column
for i in dfBlnMean.index:
    # Take the index and split it
    # print("i")
    # print(i)
    test, plat, *junk = i.split("_")
    # print(plat)
    # print(junk)
    dfFullResults.loc[i, 'Test'] = test
    dfFullResults.loc[i, 'Platform'] = plat

dfFullResults['Baseline SWV'] = baselineSWV
dfFullResults['Target SWV'] = targetSWV
# print(dfFullResults)
dfFullResults = dfFullResults.join(dfBlnMean, how='outer')
dfFullResults = dfFullResults.join(dfBlnStdDev, how='outer')
dfFullResults = dfFullResults.join(dfBlnCount, how='outer')
dfFullResults = dfFullResults.join(dfBlnSamples, how='outer')
dfFullResults = dfFullResults.join(dfBln6SigmaSamples, how='outer')
dfFullResults = dfFullResults.join(dfBlnMax, how='outer')
dfFullResults = dfFullResults.join(dfBlnMin, how='outer')
dfFullResults = dfFullResults.join(dfBln3Sigma, how='outer')
dfFullResults = dfFullResults.join(dfBlnMax3Sigma, how='outer')
dfFullResults = dfFullResults.join(dfBlnMin3Sigma, how='outer')
dfFullResults = dfFullResults.join(dfBlnCV, how='outer')
dfFullResults = dfFullResults.join(dfTarMean, how='outer')
dfFullResults = dfFullResults.join(dfTarStdDev, how='outer')
dfFullResults = dfFullResults.join(dfTarCount, how='outer')
dfFullResults = dfFullResults.join(dfTarMax, how='outer')
dfFullResults = dfFullResults.join(dfTarMin, how='outer')
dfFullResults = dfFullResults.join(dfTar3Sigma, how='outer')
dfFullResults = dfFullResults.join(dfTarMax3Sigma, how='outer')
dfFullResults = dfFullResults.join(dfTarMin3Sigma, how='outer')
dfFullResults = dfFullResults.join(dfTarCV, how='outer')
# print(dfFullResults)


# Okay now we have the basic data for the Baseline Data set, and the Target Data set
# Now need to do the 2 sample Null Hypotheses test
print("Doing null Hype Test")
nullHypeCols = ['P', 'T', 'Result', 'Pct Delta', 'Effect Size', 'Pooled Effect Size', 'Heges d', 'Used Effect Size',
                'Final Result']
dfResultsNullHype = pd.DataFrame(index=dfBlnMean.index, columns=nullHypeCols)
# print("dfResultsNullHype")
# print(dfResultsNullHype)
for i in dfBlnMean.index:
    if ((i in blnResultsData) and (i in tarResultsData)):
        # print(blnResultsData[i])
        # print(tarResultsData[i])
        # print(blnResultsData.dropna()[i].values)
        # print(tarResultsData.dropna()[i].values)
        result = scipy.stats.ttest_ind(tarResultsData[i].values, blnResultsData[i].values, nan_policy='omit')
        # print("i")
        # print(i)
        # print("result")
        # print(result)
        # print("put P result into dataframe")
        t = result[0]
        p = result[1]
        dfResultsNullHype.loc[i, 'P'] = p
        # print("put T result into dataframe")
        dfResultsNullHype.loc[i, 'T'] = t
        dfResultsNullHype.loc[i, 'Result'] = ""
        dfResultsNullHype.loc[i, 'Final Result'] = ""
        if ((dfBlnMean.loc[i, 'Bln Mean'] > 0) and (dfTarMean.loc[i, 'Tar Mean'] > 0)):
            diff = dfTarMean.loc[i, 'Tar Mean'] - dfBlnMean.loc[i, 'Bln Mean']
            dfResultsNullHype.loc[i, 'Pct Delta'] = (diff / dfBlnMean.loc[i, 'Bln Mean']) * 100
            dfResultsNullHype.loc[i, 'Final Result'] = "NC"
            if (dfResultsNullHype.loc[i, 'Pct Delta'] >= 5):
                dfResultsNullHype.loc[i, 'Final Result'] = "IMP"
            if (dfResultsNullHype.loc[i, 'Pct Delta'] <= -5):
                dfResultsNullHype.loc[i, 'Final Result'] = "FAIL"

        # Calculate simple and pooled effect sizes
        if ((dfBlnStdDev.loc[i, 'Bln Std Dev'] > 0) and (dfTarStdDev.loc[i, 'Tar Std Dev'] > 0)):
            nBln = dfBlnCount.loc[i, 'Bln Num']
            nTar = dfTarCount.loc[i, 'Tar Num']
            blnWeightedStd = (nBln - 1) * dfBlnStdDev.loc[i, 'Bln Std Dev'] ** 2
            tarWeightedStd = (nTar - 1) * dfTarStdDev.loc[i, 'Tar Std Dev'] ** 2
            if ((nBln - 1) + (nTar - 1) - 2) > 0:
                stdDevPool = math.sqrt((blnWeightedStd + tarWeightedStd) / ((nBln - 1) + (nTar - 1) - 2))
                pooled = (dfTarMean.loc[i, 'Tar Mean'] - dfBlnMean.loc[i, 'Bln Mean']) / stdDevPool
            else:
                pooled = 0

            effectSize = (dfTarMean.loc[i, 'Tar Mean'] - dfBlnMean.loc[i, 'Bln Mean']) / dfBlnStdDev.loc[
                i, 'Bln Std Dev']
            totalCount = dfBlnCount.loc[i, 'Bln Num'] + dfTarCount.loc[i, 'Tar Num']
            HegesEF = ((dfTarMean.loc[i, 'Tar Mean'] - dfBlnMean.loc[i, 'Bln Mean']) / stdDevPool) * (
                        (totalCount - 3) / (totalCount - 2.25)) * math.sqrt((totalCount - 2) / totalCount)
            if (abs(effectSize) > 1000000):
                # So if its this large just set both effect sizes to Nan
                effectSize = float('nan')
                pooled = float('nan')
                # print("dfBlnStdDev")
                # print(dfBlnStdDev[i])

            dfResultsNullHype.loc[i, 'Effect Size'] = effectSize
            dfResultsNullHype.loc[i, 'Pooled Effect Size'] = pooled
            dfResultsNullHype.loc[i, 'Heges d'] = HegesEF

            # Now record if the t-test shows a statistical significant result
            # Along with the final results based on effect size
            dfResultsNullHype.loc[i, 'Final Result'] = "NC"
            dfResultsNullHype.loc[i, 'Used Effect Size'] = effectSize
            if ((t != float("inf")) and (t != float("-inf")) and (t != float("nan"))):
                if (p <= alpha):
                    if (t < 0):
                        dfResultsNullHype.loc[i, 'Result'] = "DEG"
                        # Based on a few outcomes the check below was changed
                        if (abs(effectSize) >= 6) and (abs(pooled) >= 3):
                            dfResultsNullHype.loc[i, 'Final Result'] = "FAIL"
                            if (abs(effectSize) > abs(pooled)):
                                dfResultsNullHype.loc[i, 'Used Effect Size'] = effectSize
                            else:
                                dfResultsNullHype.loc[i, 'Used Effect Size'] = pooled
                        # Now see if the number of target results is < the number of
                        # samples that the power calculations require for a good test
                        # if not mark the results as NEN (Not Enough N)
                        if dfTarCount.loc[i, 'Tar Num'] < dfBln6SigmaSamples.loc[i, 'Tar 6Sigma Samp']:
                            dfResultsNullHype.loc[i, 'Final Result'] = "NEN"
                    else:
                        dfResultsNullHype.loc[i, 'Result'] = "IMP"
                        # Based on a few outcomes the check below was changed
                        if (abs(effectSize) >= 6) and (abs(pooled) >= 3):
                            dfResultsNullHype.loc[i, 'Final Result'] = "IMP"
                            if (abs(effectSize) > abs(pooled)):
                                dfResultsNullHype.loc[i, 'Used Effect Size'] = effectSize
                            else:
                                dfResultsNullHype.loc[i, 'Used Effect Size'] = pooled
                        # Now see if the number of target results is < the number of
                        # samples that the power calculations require for a good test
                        # if not mark the results as NEN (Not Enough N)
                        if dfTarCount.loc[i, 'Tar Num'] < dfBln6SigmaSamples.loc[i, 'Tar 6Sigma Samp']:
                            dfResultsNullHype.loc[i, 'Final Result'] = "NEN"

            # Handle the case where the effect sizes don't make sense
            if ((effectSize == float("nan")) and (pooled == float("nan"))):
                # print("NaN effect size")
                # print(i)
                if (dfResultsNullHype.loc[i, 'Pct Delta'] >= 5):
                    dfResultsNullHype.loc[i, 'Final Result'] = "IMP"
                if (dfResultsNullHype.loc[i, 'Pct Delta'] <= -5):
                    dfResultsNullHype.loc[i, 'Final Result'] = "FAIL"
                dfResultsNullHype.loc[i, 'Used Effect Size'] = dfResultsNullHype.loc[i, 'Pct Delta']
                # Now see if the number of target results is < the number of
                # samples that the power calculations require for a good test
                # if not mark the results as NEN (Not Enough N)
                if dfTarCount.loc[i, 'Tar Num'] < dfBln6SigmaSamples.loc[i, 'Tar 6Sigma Samp']:
                    dfResultsNullHype.loc[i, 'Final Result'] = "NEN"

# Now add the null hype results to the big results data
dfFullResults = dfFullResults.join(dfResultsNullHype, how='outer')
dfFullResults.dropna(axis=0, how='all', inplace=True)
pd.set_option('display.max_rows', 1000)
# print("dfFullResults")
# print(dfFullResults)
pd.set_option('display.max_rows', 10)

# Calculations are now over!!! Time to display the data
print("Create the spreadsheets")
# Now try to see if we can generate an XL spreadsheet with the data frame
if ((testType == 'FULL') or (testType == 'ALL') or (testType == 'Exec')):
    xlName = reportsPath + targetSWV + '_' + platFam + '_AllData.xlsx'
    writer = pd.ExcelWriter(xlName, engine='xlsxwriter')
    dfFullResults.to_excel(writer, sheet_name='All Data', index=False)
    workbook = writer.book
    worksheet1 = writer.sheets['All Data']
    decimals3 = workbook.add_format({'num_format': '#,##0.000'})
    decimals0 = workbook.add_format({'num_format': '#,##0'})
    worksheet1.set_column('A:B', 20)
    worksheet1.set_column('C:C', 25)
    worksheet1.set_column('D:D', 50)
    # Set this for most of the columns
    worksheet1.set_column('E:M', 20, decimals0)
    worksheet1.set_column('O:V', 20, decimals0)

    # Set this for just the smaller ones
    worksheet1.set_column('G:H', 10)
    worksheet1.set_column('N:N', 10, decimals3)
    worksheet1.set_column('W:Y', 10, decimals3)
    worksheet1.set_column('Q:Q', 10)
    worksheet1.set_column('W:Y', 10, decimals3)
    worksheet1.set_column('AA:AC', 10, decimals3)
    worksheet1.set_column('AD:AD', 15)
    worksheet1.autofilter('A1:AG1')
    writer.save()

# The first run of this would be to display the Bln and Tar CV values that are larger than 2%
if ((testType == 'CV') or (testType == 'ALL') or (testType == 'Exec')):
    # This means show just the high CV values for both the Bln and Tar software versions
    # Sort the list with the highest at the top. Only include CVs that are > 4
    highCV = dfBlnCV['Bln CV'] >= 4
    dfBlnHighCV = dfBlnCV[highCV]
    dfBlnHighCV['Baseline SWV'] = baselineSWV
    dfBlnHighCV = dfBlnHighCV.sort_values(by=['Bln CV'], ascending=False)
    highCV = dfTarCV['Tar CV'] >= 4
    dfTarHighCV = dfTarCV[highCV]
    dfTarHighCV['Target SWV'] = targetSWV
    dfTarHighCV = dfTarHighCV.sort_values(by=['Tar CV'], ascending=False)

    # Write each dataframe to a different worksheet.
    xlName = reportsPath + targetSWV + '_' + platFam + '_HighCV.xlsx'
    writer = pd.ExcelWriter(xlName, engine='xlsxwriter')
    dfBlnHighCV.to_excel(writer, sheet_name='Bln High CV')
    dfTarHighCV.to_excel(writer, sheet_name='Tar High CV')
    workbook = writer.book
    worksheet1 = writer.sheets['Bln High CV']
    format1 = workbook.add_format({'num_format': '#,##0.00'})
    worksheet1.set_column('A:A', 25)
    worksheet1.set_column('B:B', 15, format1)
    worksheet1.set_column('C:C', 50)
    worksheet1.set_zoom(120)
    worksheet2 = writer.sheets['Tar High CV']
    worksheet2.set_column('A:A', 25)
    worksheet2.set_column('B:B', 15, format1)
    worksheet2.set_column('C:C', 50)
    worksheet2.set_zoom(120)
    writer.save()

# Now do a endResults format of the data. This is what we will send to the outside
# world once we have scrubbed the data and are satisfied that the results are good
if ((testType == 'endResults') or (testType == 'ALL') or (testType == 'Exec')):
    colsToWrite = ['Test', 'Description', 'Traffic', 'Platform', 'Baseline SWV', 'Target SWV', 'Bln Mean', 'Bln 3Sigma',
                   'Tar Mean', 'Tar 3Sigma', 'Used Effect Size', 'Final Result']
    dfFail = dfFullResults['Final Result'] == 'FAIL'
    dfImp = dfFullResults['Final Result'] == 'IMP'
    dfNC = dfFullResults['Final Result'] == 'NC'
    dfendResults = dfFullResults[dfFail | dfImp | dfNC]
    # Need to map the test case name into a description of the test case so create a new data frame from the dfendResults df
    testCaseInfo = {}
    dfTestCaseMap = pd.DataFrame(index=dfendResults.index, columns=['Description', 'Traffic'])
    for i in dfendResults.index:
        # print("i")
        # print(dfendResults.loc[i,'Test'])
        testCaseInfo = testcases.find({"_id": dfendResults.loc[i, 'Test']}, {"_id": 0, "feature": 1, "traffic": 1})
        for rslt in testCaseInfo:
            # print(rslt)
            dfTestCaseMap.loc[i, 'Description'] = rslt['feature']
            dfTestCaseMap.loc[i, 'Traffic'] = rslt['traffic']
    # Now join the test case map to the end results df
    # print("dfTestCaseMap:\n",dfTestCaseMap )
    dfendResults = dfendResults.join(dfTestCaseMap, how='outer')
    # print(dfendResults)

    xlName = reportsPath + targetSWV + '_' + platFam + '_Results.xlsx'
    writer = pd.ExcelWriter(xlName, engine='xlsxwriter')
    blankDF = pd.DataFrame()
    blankDF.to_excel(writer, sheet_name='Summary')
    dfendResults.to_excel(writer, sheet_name='Results', index=False, columns=colsToWrite)
    workbook = writer.book
    format1 = workbook.add_format({'num_format': '#,##0'})
    format2 = workbook.add_format({'bold': True})
    cmdToRun = 'nullHypTestNew.py ' + baselineSWV + ' ' + targetSWV + ' ' + platFam + ' ' + testType
    worksheet1 = writer.sheets['Summary']
    worksheet1.write('A1', cmdToRun)
    worksheet1.write('B3', 'Count')
    worksheet1.write('C3', '%')
    worksheet1.write('A4', 'Failed')
    worksheet1.set_column('B:B', 15)
    worksheet1.set_column('C:C', 10, format1)
    worksheet1.write_formula('B4', '=COUNTIF(Results!L:L,"FAIL")')
    worksheet1.write('A5', 'No Change')
    worksheet1.write_formula('B5', '=COUNTIF(Results!L:L,"NC")')
    worksheet1.write('A6', 'Improvement')
    worksheet1.write_formula('B6', '=COUNTIF(Results!L:L,"IMP")')
    worksheet1.write('A7', 'Total')
    worksheet1.write_formula('B7', '=SUM(B4:B6)')
    worksheet1.write_formula('C4', '=100*(B4/$B$7)')
    worksheet1.write_formula('C5', '=100*(B5/$B$7)')
    worksheet1.write_formula('C6', '=100*(B6/$B$7)')
    worksheet2 = writer.sheets['Results']
    worksheet2.set_column('A:A', 15)
    worksheet2.set_column('B:B', 25)
    worksheet2.set_column('C:C', 15)
    worksheet2.set_column('D:D', 30)
    worksheet2.set_column('E:J', 15, format1)
    worksheet2.set_column('K:K', 10, format1)
    worksheet2.set_column('L:L', 15, format2)
    worksheet2.autofilter('A1:L1')
    worksheet2.set_zoom(120)

    # Need to adjust some column headings
    worksheet2.write('G1', 'Baseline Mean', format2)
    worksheet2.write('H1', 'Baseline Range (+/-)', format2)
    worksheet2.write('I1', 'Target Mean', format2)
    worksheet2.write('J1', 'Target Range (+/-)', format2)
    worksheet2.write('K1', 'Used Effect Size', format2)
    worksheet2.write('L1', 'Final Result', format2)
    writer.save()

    # Now write a CSV file that contains just the FAILed tests
    colsToWriteCSV = ['Test', 'Platform', 'Baseline SWV', 'Target SWV', 'Bln Mean', 'Tar Mean', 'Used Effect Size',
                      'Final Result']
    csvName = reportsPath + targetSWV + '_' + platFam + '_' + 'FAIL_Results.csv'
    dfCSV = dfFullResults[dfFail]
    dfCSV.to_csv(csvName, index=False, columns=colsToWriteCSV)

# Now do a Marketing format of the data. This is what we will send to the outside
# world once we have scrubbed the data and are satisfied that the results are good
if ((testType == 'Marketing') or (testType == 'ALL') or (testType == 'Exec')):
    colsToWrite = ['Test', 'Description', 'Traffic', 'Platform', 'Baseline SWV', 'Bln Mean', 'Bln 3Sigma']
    # Need to map the test case name into a description of the test case so create a new data frame from the dfendResults df
    testCaseInfo = {}
    dfTestCaseMap = pd.DataFrame(index=dfFullResults.index, columns=['Description', 'Traffic'])
    for i in dfFullResults.index:
        # print("i")
        # print(dfFullResults.loc[i,'Test'])
        testCaseInfo = testcases.find({"_id": dfFullResults.loc[i, 'Test']}, {"_id": 0, "feature": 1, "traffic": 1})
        for rslt in testCaseInfo:
            # print(rslt)
            dfTestCaseMap.loc[i, 'Description'] = rslt['feature']
            dfTestCaseMap.loc[i, 'Traffic'] = rslt['traffic']
    # Now join the test case map to the end results df
    # print("dfTestCaseMap:\n",dfTestCaseMap )
    dfFullResults = dfFullResults.join(dfTestCaseMap, how='outer')
    # print(dfFullResults)

    xlName = reportsPath + targetSWV + '_' + platFam + '_Marketing.xlsx'
    writer = pd.ExcelWriter(xlName, engine='xlsxwriter')
    dfFullResults.to_excel(writer, sheet_name='Results', index=False, columns=colsToWrite)
    workbook = writer.book
    format1 = workbook.add_format({'num_format': '#,##0'})
    format2 = workbook.add_format({'bold': True})
    worksheet = writer.sheets['Results']
    worksheet.set_column('A:A', 15)
    worksheet.set_column('B:B', 25)
    worksheet.set_column('C:C', 15)
    worksheet.set_column('D:D', 30)
    worksheet.set_column('E:G', 15, format1)
    worksheet.autofilter('A1:G1')
    worksheet.set_zoom(120)

    # Need to adjust some column headings
    worksheet.write('F1', 'Bln Mean', format2)
    worksheet.write('G1', 'Bln Range (+/-)', format2)
    writer.save()
