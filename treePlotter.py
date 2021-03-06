import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy = parentPt, xycoords = 'axes fraction', xytext = centerPt, textcoords = 'axes fraction', va = 'center', ha = 'center',bbox = nodeType,arrowprops = arrow_args)


def createPlot():
    fig = plt.figure(1,facecolor = 'white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('decision', (0.5, 0.1),(0.1, 0.5),decisionNode)
    plotNode('leaf', (0.8, 0.1),(0.1, 0.8),leafNode)
    plt.show()

def getNumLeafs(myTree):
	'''
	To obtain width of the tree for ploting
	''' 
	firstKey = myTree.keys()[0]
	subTree = myTree[firstKey]
	leafNum = 0
	for key in subTree.keys():
		if type(subTree[key]).__name__ == 'dict':
			leafNum += getNumLeafs(subTree[key])
		else:
			leafNum += 1

	return leafNum

def getTreeDepth(myTree):
	'''
	To obtain depth of the tree for ploting
	'''
	firstKey = myTree.keys()[0]	
	subTree = myTree[firstKey]
	maxDepth = 1
	for key in subTree.keys():
		curDepth = 0
		if type(subTree[key]).__name__ == 'dict':
			curDepth = getTreeDepth(subTree[key])
		maxDepth = max(maxDepth,curDepth+1)
	return maxDepth

def retrieveTree(i):
	'''
	To form some examples for testing
	'''

	listOfTrees = [{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}}
			,{'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
	return listOfTrees[i]


def plotMidText(cntrPt,parentPt,txtString):
	'''
	Indicate the relationship between parentNode and childNode
	'''
	midx = (parentPt[0] - cntrPt[0])/2 + cntrPt[0]
	midy = (parentPt[1] - cntrPt[1])/2 + cntrPt[1]
	createPlot.ax1.text(midx,midy,txtString)


def 
