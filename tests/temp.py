
import copy
class Spool:
	"""
	New class for spool, for 'flocking' behavior
	"""

	def __init__(self):
		self.threads = []
		self.t = None
		self.blob_dist_thresh = 9
	def reel(self, positions, t=0):

		# if no threads already exist, add all incoming points to new threads them and positions tracker
		if self.threads == []:
			for i in range(len(positions)):
				self.threads.append(Thread(positions[i]))
			self.update_positions()
			self.t = 0
		# if threads already exist
		else:
			# match points based on a max-threshold euclidean distance based matching
			diff = scipy.spatial.distance.cdist(self.positions, positions, metric='euclidean')
			matchings, unmatched, newpoints = calc_match(diff, self.blob_dist_thresh)

			dvec = np.zeros(self.threads[0].get_position_mostrecent.shape)
			for match in matchings:
				dvec += self.threads[match[0]].get_position_mostrecent()-positions[match[1]]
				self.threads[match[0]].update_position(positions[match[1]], t = self.t)
			dvec *= 1/len(matchings)

			for point in newpoints:
				self.threads.append(Thread(positions[point], t=self.t))
			self.t += 1
			self.update_positions()
			self.update_predictions()


	def update_positions(self):
		"""
		updates positions based on threads of matched points
		"""
		self.positions = np.zeros((len(self.threads), self.threads[0].get_position_mostrecent().shape[0]))

		for i in range(len(self.threads)):
			self.positions[i] = self.threads[i].get_position_mostrecent()


	def update_predictions(self):
		self.predictions = np.zeros((len(self.threads), self.threads[0].get_position_mostrecent().shape[0]))

		for i in range(len(self.threads)):
			self.predictions[i] = 2*self.threads[i].get_position_t(self.threads[i].t[-1])-self.threads[i].get_position_t(self.threads[i].t[-2])










