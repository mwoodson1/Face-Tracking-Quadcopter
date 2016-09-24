'''
Training:
	For each training video:
		-construct a sequence of 1D-distance signals from person centroids

	Perform PCA on the sequence to reduce dimensionality

Test Pre-process:
	Given a training sequence
		-construct the sequence of 1D-distance signals
		-project onto eigenspace found in training process

Classification:
	Perform KNN, SVM, or any other classification algorithm we would like to try
'''
import os, pickle, cv2
import numpy as np
from numpy import linalg as LA
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV

def denoiseSilhouette(img):
	'''
	Denoise the image silhouettes to give a more rounded shape. Use this
	when using real world images as background model will not be given
	and silhouettes will likely be noisier than training data.
	'''
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
	erosion = cv2.erode(img,kernel,iterations = 2)
	dilation = cv2.dilate(erosion,kernel,iterations = 3)
	closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	return closing

def findMaxContour(img):
	'''
	Find the largest contour in the image which should be the outline
	of the person.
	'''
	img2, contour,hier = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	areas = [cv2.contourArea(c) for c in contour]
	cnt=contour[np.argmax(areas)]

	M = cv2.moments(cnt)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return cx, cy, cnt

def toDistSeq(img_path):
	'''
	Given the path to a folder of images representing frames of a video,
	turn each frame into a 1D distance signal and concatenate into an
	image.
	'''
	img = cv2.imread(img_path,0)
	denoised = denoiseSilhouette(img)
	cx, cy, cnt = findMaxContour(denoised)
	dists = np.zeros(len(cnt))

	#Find distance betweem center to all contour poiints
	for i in xrange(len(cnt)):
		x = cnt[i][0][0]
		y = cnt[i][0][1]
		dists[i] = np.sqrt(((x-cx)**2)+((y-cy)**2))

	#L1 normalization of distances
	dists_norm = dists / LA.norm(dists,1)

	#Subsample the vector to a constant size
	indices = np.linspace(0,len(dists_norm),360,endpoint=False,dtype=np.int32)
	sampled_dists = dists_norm[indices]
	return sampled_dists

def loadData(datapath,dataset):
	'''
	MOBO path form:
	/data/moboBgSub/userID/mode/vidID/orientation/frame.pbm

	CASIA path form:
	/DatasetB/silhouettes/subjectID/walkingStatus-seqNum/viewAngle

	CASIA has more data to train on so it's reccomended to use that dataset when training
	'''
	N = 100   #The number of frames from each sequence to use
	if(dataset=="MOBO"):
		orientation_map = {'vr03_7':0,'vr05_7':1,'vr07_7':2,
						   'vr13_7':3,'vr16_7':4,'vr17_7':5,}
		X = np.zeros((6,100*25,360))
		y = np.zeros((6,100*25))
		topLevel = datapath+"/moboBgSub/"
		indices = [0,0,0,0,0,0]
		for userID in sorted(next(os.walk(topLevel))[1]):
			mode = "slowWalk"
			for orientation in sorted(next(os.walk(topLevel+userID+"/"+"slowWalk"))[1]):
				n = 0
				clf_index = orientation_map[orientation]
				for img in sorted(next(os.walk(topLevel+userID+"/"+mode+"/"+orientation))[2]):
					if(n >= N):
						break
					img_path = topLevel+userID+"/"+mode+"/"+orientation+"/"+img
					distSignal = toDistSeq(img_path)
					X[clf_index,indices[clf_index],:] = distSignal
					y[clf_index,indices[clf_index]] = userID
					indices[clf_index] += 1
					n += 1
		return X, y

	elif(dataset=="CASIA"):
		topLevel = datapath+"/silhouettes/"
		for subID in next(os.walk(topLevel))[1]:
			for seqNum in next(os.walk(topLevel+subID))[1]:
				for viewAngle in next(os.walk(topLevel+subID+"/"+seqNum))[1]:
					for img in next(os.walk(topLevel+subID+"/"+seqNum+"/"+viewAngle))[2]:
						img_path = topLevel+subID+"/"+seqNum+"/"+viewAngle+"/"+img
						distSignal = toDistSeq(img_path)
	else:
		print "Invalid dataset input"
		return 0

def gridSearch(X_train,y_train,angle):
	"""
	Performs a grid search to find the best classifier hyperparameters using LDA
	with a KNN classifier.
	"""
	component_grid = [5,10,20,50,75,100]
	neighbor_grid = [2,3,4,5,6,7,9,11,15,20,25,30,40]

	estimators = [('reduce_dim', LDA(solver='eigen')), ('knn', KNeighborsClassifier())]
	clf = Pipeline(estimators)

	params = {'reduce_dim__n_components':neighbor_grid,
	          'knn__n_neighbors':component_grid}

	grid_search = GridSearchCV(clf,param_grid=params)
	grid_search.fit(X_train,y_train)

	pickle.dump(grid_search,open("model"+str(angle)+".p","wb"))
	return grid_search

def trainAngleClf(X,y,angle):
	#Classifier for angle1
	print "Training classifier for angle "+str(angle)+"..."
	X = X[angle,:,:]
	y = y[angle,:]

	#Create train and test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#Check if model is already cached
	if(os.path.isfile("model"+str(angle)+".p")):
		print("Loading cached model for angle"+str(angle)+"...")
		clf = pickle.load(open("model"+str(angle)+".p","rb"))
	else:
		clf = gridSearch(X_train,y_train,angle)

	print clf.score(X_test,y_test)
	return clf

#Load data
print("Loading data...")
X, y = loadData('data','MOBO')
le = preprocessing.LabelEncoder()
le.fit(np.unique(y))
y = le.transform(y)

clf0 = trainAngleClf(X,y,0)
clf1 = trainAngleClf(X,y,1)
clf2 = trainAngleClf(X,y,2)
clf3 = trainAngleClf(X,y,3)
clf4 = trainAngleClf(X,y,4)
clf5 = trainAngleClf(X,y,5)
