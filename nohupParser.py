import collections

def main():
   formattedData = loadData('nohupCopy.out')
   pointDictList = []
   accuracyList = []
   lossList = []
   dictEntry = collections.defaultdict(float)
   accuracyRun = []
   lossRun = []
   for i in range(len(formattedData)):
	if formattedData[i] == '================':
            #watch for python pass by reference/value issuesi
	    if dictEntry:
		totalPoints = dictEntry['truePos'] + dictEntry['falsePos'] + dictEntry['trueNeg'] + dictEntry['falseNeg']
		for key, value in dictEntry.iteritems():
		    if key == 'eta' or key == 'NumNextToEta':
			pass
		    #if key != 'eta' or key != 'NumNextToEta':
		    else:
			dictEntry[key] = (value)/totalPoints
	        pointDictList.append(dictEntry)
	    if accuracyRun:
		accuracyList.append(accuracyRun)
	    if lossRun:
		lossList.append(lossRun)
 	    dictEntry = collections.defaultdict(float)
	    accuracyRun = []
	    lossRun = []
            etaNum = formattedData[i+1].split()
	    dictEntry['eta'] = etaNum[0]
	    dictEntry['NumNextToEta'] = etaNum[1]
    	if 'total datapoints' in formattedData[i]:
            truePosLine = formattedData[i+1].split()
   	    dictEntry['truePos'] += int(truePosLine[1])
  
	    trueNegLine = formattedData[i+2].split()
	    dictEntry['trueNeg'] += int(trueNegLine[1])
	    
	    falsePosLine = formattedData[i+3].split()
            dictEntry['falsePos'] += int(falsePosLine[1])
	    
	    falseNegLine = formattedData[i+4].split()
            dictEntry['falseNeg'] += int(falseNegLine[1])
		
	    accuracyLine = formattedData[i+5].split()
            accuracyRun.append(float(accuracyLine[1]))
	
	    avgLossLine = formattedData[i+6].split()
	    lossRun.append(float(avgLossLine[1]))
   print pointDictList
   for accuracy in accuracyList:
       print accuracy
   print lossList

              

def loadData(fileName):
    with open(fileName) as f:
        data = f.readlines()
        formattedData = []
	for i in range(len(data)):
    	    formattedData.append(data[i].strip('\n'))
	return formattedData

main()
