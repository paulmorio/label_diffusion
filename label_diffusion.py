#!/usr/bin/env python

"""
Python Implementation of Label Diffusion Algorithm as Described in 
Zhou et. al. "Learning with Local and Global Consistency"

Author: Paul Scherer
Date: 19/02/2018
"""

import networkx as nx
import numpy as np

def label_diffusion(adj, known_labels, known_nodes, unknown_nodes, alpha=0.99, precision=1e20, max_iterations=100):
	"""
	Function implementing the whole label diffusion algorithm, with and without regularisation as referred in
	Zhou et al. 'Learning with Local and Global Consistency'. Mainly to help out those poor folks who cant find 
	implementations of this algorithm online.

	Keyword Arguments:
	adj -- an adjacency matrix of the network
	known_labels -- an mxk matrix where m is the number of nodes whose label is known, k is the number of labels
					this is a one hot indication of label
	known_nodes -- vector of size m, contains the indices of nodes whose labels are known
	unknown_nodes -- vector of nodes indices whose labels are to be predicted
	alpha -- propagation parameter (default 0.99)
	precision -- level of precision required before stopping iterations (default 1e8)
	max_iterations -- maximum number of iterations (default 100)

	Output: "predictions_score", matrix whose element ij is the predicted score of label j for the ith node to predict.
			"predictions" vector of predictions for evaluation
	"""

	# Check adj matrix is square
	n, m = adj.shape
	if n != m:
		print "The adjacency matrix provided isnt square"
		return None

	# Check alpha is in ]0,1[
	if (alpha <= 0 or alpha >= 1):
		print "Alpha should be between 0 and 1 exclusive"
		return None

	# Point 1 is given by adj
	# Normalize the adjaceny matrix (point 2 in paper)
	D = (np.power(np.sum(adj, 1), -0.5))
	D = np.squeeze(np.asarray(D))
	D = np.diag(D)
	diffusion_matrix = D*adj*D

	k = known_labels.shape[1]
	initial_scores = np.zeros((n,k))
	# known nodes is the mask
	initial_scores[known_nodes, :] = known_labels
	predictions_score = initial_scores

	# # Propagation
	# for i in range(max_iterations):
	# 	last_score = predictions_score
	# 	predictions_score = (alpha * diffusion_matrix * predictions_score) + ((1-alpha)*initial_scores)
	# 	print "Iteration %s Completed" % (i)
	# 	if np.max(np.max(np.abs(last_score - predictions_score))) > precision:
	# 		print "Precision Required Reached in iteration %s " % (i)
	# 		break
	predictions_score = (np.linalg.inv(np.identity(n) - alpha*diffusion_matrix)) * predictions_score

	# return list of label predictions
	predictions = []
	for onehot in predictions_score:
		predictions.append(np.argmax(onehot))

	return predictions_score, predictions

def evaluation(predictions, test_node_indices, y_test):
	pred = [] # gotta make sure we're taking the predictions for the test data
	for i in test_node_indices:
		pred.append(predictions[i])

	pred = np.array(pred)

	return (np.mean(np.equal(y_test, pred)))

def label_diffusion_features(adj, features, known_labels, known_nodes, unknown_nodes, alpha=0.99, precision=1e20, max_iterations=100):
	"""
	Function implementing the whole label diffusion algorithm, with and without regularisation as referred in
	Zhou et al. 'Learning with Local and Global Consistency'.This version incorporates the features of nodes. 

	Keyword Arguments:
	adj -- an adjacency matrix of the network
	features -- the features matrix nxp, where n is the number of nodes in the graph, p the dimension of the 
	known_labels -- an mxk matrix where m is the number of nodes whose label is known, k is the number of labels
					this is a one hot indication of label
	known_nodes -- vector of size m, contains the indices of nodes whose labels are known
	unknown_nodes -- vector of nodes indices whose labels are to be predicted
	alpha -- propagation parameter (default 0.99)
	precision -- level of precision required before stopping iterations (default 1e8)
	max_iterations -- maximum number of iterations (default 100)

	Output: "predictions_score", matrix whose element ij is the predicted score of label j for the ith node to predict.
			"predictions" vector of predictions for evaluation
	"""

	# Create the Gaussian affinity matrix w
	w =  np.zeros(adj.shape)
	sigmas = np.zeros(adj.shape)
	adjn, adjm = adj.shape

	# to help calculate sigma in affinity matrix
	for row in range(0,adjn):
		if row % 5 == 0:
			print "sigmas " + str(row)
		for column in range(0,adjm):
			if row == column:
				sigmas[row, column] = 0
			else:
				xi = features[row].todense()
				xj = features[column].todense()
				dist = np.linalg.norm(xi-xj)
				sigmas[row, column] = dist
	sigma = (np.max(np.max(sigmas)))/np.power(adjn,(1/features.shape[1]))
	# as recommended by moufsyet

	for row in range(0,adjn):
		if row % 5 == 0:
			print row
		for column in range(0,adjm):
			# if row == column:
			# 	w[row, column] = 0
			# else:
			# 	dist = sigmas[row, column] # little bit a memoization 
			# 	w[row, column] = np.exp((-np.power(dist,2))/(2*sigma))
			if adj[row,column] == 1:
				dist = sigmas[row, column] # little bit a memoization 
				w[row, column] = np.exp((-np.power(dist,2))/(2*sigma))
			else:
				w[row, column] = 0				

	w = np.matrix(w)
	# Check affinitity matrix w is square
	n, m = w.shape
	print n,m
	print adj.shape
	print type(adj)
	print type(w)
	print w
	#lapl = 
	if n != m:
		print "The affinity matrix provided isnt square"
		return None

	# Check alpha is in ]0,1[
	if (alpha <= 0 or alpha >= 1):
		print "Alpha should be between 0 and 1 exclusive"
		return None

	# Point 1 is given by w
	# Normalize the adjaceny matrix (point 2 in paper)
	# zhou vs me.
	D = (np.power(np.sum(adj, 1), -0.5))
	D = np.squeeze(np.asarray(D))
	D = np.diag(D) # this is the degree matrix
	laplacian = D - w
	diffusion_matrix = D*laplacian*D

	k = known_labels.shape[1]
	initial_scores = np.zeros((n,k))
	# known nodes is the mask
	initial_scores[known_nodes, :] = known_labels
	predictions_score = initial_scores

	# # Propagation
	# for i in range(max_iterations):
	# 	last_score = predictions_score
	# 	predictions_score = (alpha * diffusion_matrix * predictions_score) + ((1-alpha)*initial_scores)
	# 	print "Iteration %s Completed" % (i)
	# 	if np.max(np.max(np.abs(last_score - predictions_score))) > precision:
	# 		print "Precision Required Reached in iteration %s " % (i)
	# 		#break
	# instead of propagation
	predictions_score = (np.linalg.inv(np.identity(n) - alpha*diffusion_matrix)) * predictions_score

	# return list of label predictions
	predictions = []
	for onehot in predictions_score:
		predictions.append(np.argmax(onehot))

	return predictions_score, predictions


def label_diffusion_zhou(adj, features, known_labels, known_nodes, unknown_nodes, alpha=0.99, precision=1e20, max_iterations=100):
	"""
	Function implementing the whole label diffusion algorithm, with and without regularisation as referred in
	Zhou et al. 'Learning with Local and Global Consistency'.This version incorporates the features of nodes. 

	Keyword Arguments:
	adj -- an adjacency matrix of the network
	features -- the features matrix nxp, where n is the number of nodes in the graph, p the dimension of the 
	known_labels -- an mxk matrix where m is the number of nodes whose label is known, k is the number of labels
					this is a one hot indication of label
	known_nodes -- vector of size m, contains the indices of nodes whose labels are known
	unknown_nodes -- vector of nodes indices whose labels are to be predicted
	alpha -- propagation parameter (default 0.99)
	precision -- level of precision required before stopping iterations (default 1e8)
	max_iterations -- maximum number of iterations (default 100)

	Output: "predictions_score", matrix whose element ij is the predicted score of label j for the ith node to predict.
			"predictions" vector of predictions for evaluation
	"""

	# Create the Gaussian affinity matrix w
	w =  np.zeros(adj.shape)
	sigmas = np.zeros(adj.shape)
	adjn, adjm = adj.shape

	# to help calculate sigma in affinity matrix
	for row in range(0,adjn):
		if row % 5 == 0:
			print "sigmas " + str(row)
		for column in range(0,adjm):
			if row == column:
				sigmas[row, column] = 0
			else:
				xi = features[row].todense()
				xj = features[column].todense()
				dist = np.linalg.norm(xi-xj)
				sigmas[row, column] = dist
	sigma = (np.max(np.max(sigmas)))/np.power(adjn,(1/features.shape[1]))
	# as recommended by moufsyet

	for row in range(0,adjn):
		if row % 5 == 0:
			print row
		for column in range(0,adjm):
			if row == column:
				w[row, column] = 0
			else:
				dist = sigmas[row, column] # little bit a memoization 
				w[row, column] = np.exp((-np.power(dist,2))/(2*sigma))	

	w = np.matrix(w)
	# Check affinitity matrix w is square
	n, m = w.shape
	print n,m
	print adj.shape
	print type(adj)
	print type(w)
	print w
	#lapl = 
	if n != m:
		print "The affinity matrix provided isnt square"
		return None

	# Check alpha is in ]0,1[
	if (alpha <= 0 or alpha >= 1):
		print "Alpha should be between 0 and 1 exclusive"
		return None

	# Point 1 is given by w
	# Normalize the adjaceny matrix (point 2 in paper)
	# zhou vs me.
	D = (np.power(np.sum(w, 1), -0.5))
	D = np.squeeze(np.asarray(D))
	D = np.diag(D) # this is the degree matrix
	diffusion_matrix = D*w*D

	k = known_labels.shape[1]
	initial_scores = np.zeros((n,k))
	# known nodes is the mask
	initial_scores[known_nodes, :] = known_labels
	predictions_score = initial_scores

	# # Propagation
	# for i in range(max_iterations):
	# 	last_score = predictions_score
	# 	predictions_score = (alpha * diffusion_matrix * predictions_score) + ((1-alpha)*initial_scores)
	# 	print "Iteration %s Completed" % (i)
	# 	if np.max(np.max(np.abs(last_score - predictions_score))) > precision:
	# 		print "Precision Required Reached in iteration %s " % (i)
	# 		#break
	# instead of propagation
	predictions_score = (np.linalg.inv(np.identity(n) - alpha*diffusion_matrix)) * predictions_score

	# return list of label predictions
	predictions = []
	for onehot in predictions_score:
		predictions.append(np.argmax(onehot))

	return predictions_score, predictions
