import gym
import itertools
import numpy as np
import networkx as nx
import pandas as pd
from or_gym.utils import assign_env_config
from collections import deque
import random

class BloodSupplyChain(gym.Env):
	def __init__(self, *args, **kwargs):
		self._max_rewards = 20000 # change this?
		self.max_episode_steps = 3
		self.backlog = False # don't backlog unfulfilled orders
		self.alpha = 1.00 # account for time value of money
		self.seed_int = 42 
		self.sample_path = {(1,0): False} # doesn't make a difference
		self.num_products = 8 

		# add environment configuration dictionary and keyword arguments
		assign_env_config(self, kwargs)
		self.num_centers = len(self.blood_centers)
		self.num_hospitals = len(self.hospitals)
		self.bc_name = self.blood_centers['Name']

		# create graph
		self.graph = nx.DiGraph()
		#Initialize Blood Center Nodes in Graph (Supplier)

		# Node Attribute Values

		# name: Blood Center Name
		# IO: Initial Inventory of the Blood Center/ Number of donations on the first day 
		# C: Inventory Capacity of the Blood Center
		# h: Unit holding Cost for excess on-hand inventory

		# check C as inventory capacity
		for i, value in enumerate(self.blood_centers['Name']): # removed indices
			self.graph.add_nodes_from([i], name=value, 
			                          I0 = list(self.donation_data.iloc[i,7:]),
																C = self.inv_capacity[value], 
																h = 1) # remove holding cost for inventory?
			
		#Initialize Hospital Nodes in Graph (Retail)
		'''
		Node Attribute Values

		name: Hospital Name
		I0: Initial inventory available at the hospital of each blood type 
		I: Hospital inventory capacity
		'''
		for i, value in enumerate(self.hospitals['Hospital']):
			self.graph.add_nodes_from([i+self.num_centers], name=value,
			                          I0 = np.zeros(self.num_products),
																I = self.hospital_capacity[value])
			
		# Initialize Patient Request Nodes in Graph (Retail)
		'''
		Node Attribute Values

		name: Hospital Name - Patient
		'''
		for i, value in enumerate(self.hospitals['Hospital']):
			self.graph.add_nodes_from([i+self.num_centers+self.num_hospitals], 
			                          name=value+'-Patient')
			
		#Initialize edges between Blood Centers and Hospitals

		'''
		Edge Attribute Values

		p: Unit price to send materials between blood center i and hospital j
		   r: Vehicle transportation cost per minute = 34.32
			 V: Maximum distribution vehicle capacity = 50
			 h: handling cost per blood bag transported. = 16.24

			 r/V = 0.7

		L: lead time in between adjacent nodes 
		b: Cost of unfulfilled demand
		demand_dist: demand distribution for (blood_center, hospital) edge 
		dist_param: value for parameter fed to statistical distribution
		'''
		for i in range(self.num_centers):   
			for j in range(self.num_hospitals):
				self.graph.add_edges_from([(i, j+self.num_centers,
				                            {'p': 0.7*int(self.travel_times.iloc[i,j+1])+16.24 ,
																		'L': self.travel_times.iloc[i,j+1]})])

		#Initialize edges between Hospitals and Patients
		# change demand distribution parameter
		'''
		Edge Attribute Values

		p: Unit price to deliver blood to patients
		b: Cost of unfulfilled demand
		'''
		for i in range(self.num_hospitals):
			self.graph.add_edges_from([(i+self.num_centers, 
			                            i+self.num_centers+self.num_hospitals, 
																	 {'p': 0,
																	  'b': 110})])

		# Save user_D to graph metadata
		for link in self.user_D.keys():
			d = self.user_D[link]
			self.graph.edges[link]['user_D'] = d
			
		self.market = [j for j in self.graph.nodes() if len(list(self.graph.successors(j))) == 0]
		self.retail = [j for j in self.graph.nodes() if len(set.intersection(set(self.graph.successors(j)), set(self.market))) > 0]
		self.distrib = [j for j in self.graph.nodes() if 'C' in self.graph.nodes[j]]
		
		self.main_nodes = np.sort(self.distrib + self.retail)
		self.reorder_links = [e for e in self.graph.edges() if 'L' in self.graph.edges[e]] #exclude links to markets (these cannot have lead time 'L')
		self.retail_links = [e for e in self.graph.edges() if 'L' not in self.graph.edges[e]] #links joining retailers to markets
		self.network_links = [e for e in self.graph.edges()] #all links involved in sale in the network
		self.seed(self.seed_int)
	
		# action space (reorder quantities for each node for each supplier; list)
		# An action is defined for every node
		self.num_reorder_links = len(self.reorder_links) 
		###check if lt_max needed

		# ADJUST THESE VALUES
		self.lt_max = np.max([self.graph.edges[e]['L'] for e in self.graph.edges() if 'L' in self.graph.edges[e]])
		self.init_inv_max = np.max([sum(self.graph.nodes[j]['I0']) for j in self.graph.nodes() if 'I0' in self.graph.nodes[j]])
		self.capacity_max = np.max([self.graph.nodes[j]['C'] for j in self.graph.nodes() if 'C' in self.graph.nodes[j]])

		# self.pipeline_length = sum([self.graph.edges[e]['L']
		# 	for e in self.graph.edges() if 'L' in self.graph.edges[e]])

		# is this used? -- no
		# self.lead_times = {e: self.graph.edges[e]['L'] 
		# 	for e in self.graph.edges() if 'L' in self.graph.edges[e]}

		# self.pipeline_length = len(self.main_nodes)*(self.lt_max+1)
		
		self.action_space = gym.spaces.Box(
			low=np.zeros(self.num_reorder_links*self.num_products),
			# what are the maximum action values?
			high=np.ones(self.num_reorder_links*self.num_products)*(self.init_inv_max + self.capacity_max*self.max_episode_steps)*(self.num_products), 
			dtype=np.float32)
	
		'''
		Observation space: blood center inventory, hospital inventory, 
		previous demand, what else?
		'''
		self.obs_dim = (len(self.main_nodes) + len(self.retail_links))*self.num_products
		# observation space (total inventory at each node, which is any integer value)
		self.observation_space = gym.spaces.Box(
			low=np.ones(self.obs_dim)*np.iinfo(np.int32).min,
			high=np.ones(self.obs_dim)*np.iinfo(np.int32).max,
			dtype=np.float32)

		# intialize
		self.reset()
	
	def seed(self,seed=None):
			'''
			Set random number generation seed
			'''
		# seed random state
			if seed != None:
				np.random.seed(seed=seed)
			
	
	def _RESET(self):
		'''
		Create and initialize all variables and containers.
		Nomenclature:
		NOT CORRECT
			I = On hand inventory at the start of each period at each stage (except last one).
			T = Number of periods.
			R = Replenishment order placed at each period at each stage (except last one).
			D = Customer demand at each period (at the retailer)
			S = Sales performed at each period at each stage.
			B = Backlog at each period at each stage.
			LS = Lost sales at each period at each stage.
			P = Total profit at each stage.
		'''

		# not correct according to definition
		T = self.max_episode_steps
		#Num of blood centers
		J = len(self.main_nodes)
		#Number of retail-market pairs
		RM = len(self.retail_links)  # number of retailer-market pairs
		#Combination of hospitals to blood centers
		PS = len(self.reorder_links) # number of purchaser-supplier pairs in the network
		SL = len(self.network_links) # number of edges in the network (excluding links form raw material nodes)
		
		# simulation result lists
		self.X = pd.DataFrame(data = [[list(np.zeros(self.num_products)) for _ in range(J)] for _ in range(self.max_episode_steps)],
							columns = self.main_nodes)  # inventory at the beginning of each period
		
			# Add demand center inventory
			# for rm in self.retail:

		# self.Y = pd.DataFrame(data=[[list(np.zeros(self.num_products)) for _ in range(PS)] for _ in range(T+1)], 
		# 					columns = pd.MultiIndex.from_tuples(self.reorder_links,
		# 					names = ['Source','Receiver'])) # pipeline inventory at the beginning of each period
	
		self.R = pd.DataFrame(data=[[list(np.zeros(self.num_products)) for _ in range(PS)] for _ in range(T)],
				 columns=pd.MultiIndex.from_tuples(self.reorder_links, names=['Supplier', 'Requester'])) # replenishment orders

		# what is this?
		self.S = pd.DataFrame(data = [[list(np.zeros(self.num_products)) for _ in range(SL)] for _ in range(T)], 
							columns = pd.MultiIndex.from_tuples(self.network_links,
							names = ['Seller','Purchaser'])) # units sold

		self.D=pd.DataFrame(data = [[list(np.zeros(self.num_products)) for _ in range(RM)] for _ in range(T)],
							columns = pd.MultiIndex.from_tuples(self.retail_links, 
							names = ['Retailer','Market'])) # demand at retailers
	
		self.U=pd.DataFrame(data = [[list(np.zeros(self.num_products)) for _ in range(RM)] for _ in range(T)],
							columns = pd.MultiIndex.from_tuples(self.retail_links, 
							names = ['Retailer','Market'])) # unfulfilled demand for each market - retailer pair
		
		self.P=pd.DataFrame(data = np.zeros([T, J]), 
							columns = self.main_nodes) # profit at each node
		
		# initialization
		for j in self.main_nodes:
			self.X.loc[0,j]=self.graph.nodes[j]['I0']

		self.period = 0 # initialize time
		# self.Y.loc[0,:]=np.zeros(PS) # initial pipeline inventory

		# Make this by product
		self.action_log = np.zeros([T, PS], dtype=np.float32)

		# set state
		self._update_state()
		
		return self.state

	def _update_state(self):
		# State is a concatenation of demand and inventory at each time step
		# demand is an array with 8*147 values, demand for each blood product in each hospital
		
		demand = np.hstack([self.D[d].iloc[self.period] for d in self.retail_links])
		inventory = np.hstack([self.X[n].iloc[self.period] for n in self.main_nodes])
	
		self.state = np.hstack([demand, inventory])
	
	def _STEP(self, action):

		t = self.period
		if type(action) != dict: # convert to dict if a list was given
			action = {key: action[i:i+self.num_products] for i, key in enumerate(self.reorder_links)}

		# Place Orders
	  # get minimum for each blood product between request and inventory
		for key in action.keys():
			get_min = []

			supplier = key[0]
			purchaser = key[1]
			X_supplier = self.X.loc[t,supplier] # request limited by available inventory at beginning of period
			for i in range(self.num_products):    
				request = np.float32(round(action[key][i]))
				request = abs(request)
				# if (request<0 | X_supplier[i])<0:
				#     print("X_supplier: {}".format(X_supplier[i]), "Request: {}".format(request), "Supplier: {}".format(supplier), "Purchaser: {}".format(purchaser),"time: {}".format(t))
				# get_min.append(max(min(request, X_supplier[i]),0))
				get_min.append(min(request, abs(X_supplier[i])))
# 			print("R/get_min: {}".format(get_min))
		
			self.R.loc[t,(supplier, purchaser)] = get_min

			# what's the point of updating this 
			self.S.loc[t,(supplier, purchaser)] = get_min
			
		for j in self.distrib:
			incoming_data = self.donation_data[self.donation_data['hospital'] == self.bc_name[j]].iloc[t, 7:]
			outgoing = []

			# v can be changed to indicate how much passes testing from donations
			if 'v' in self.graph.nodes[j]: #extract production yield
					v = self.graph.nodes[j]['v']
			else:
					v = 1
			for k in self.graph.successors(j):
					outgoing += [self.S.loc[t,(j,k)]]

			outgoing = 1/v * np.array(outgoing).reshape(self.num_products, -1)
# 			self.X.loc[t+1,j] = np.array(self.X.loc[t,j]) + incoming_data.values - np.sum(outgoing, axis=1)
	 
		#Receive deliveries and update inventories
		for j in self.retail:
			outgoing = []
			for k in self.graph.successors(j):
					outgoing += [self.S.loc[t,(j,k)]]

			if np.sum(self.X.loc[t,j]) - np.sum(outgoing) < self.graph.nodes[supplier]['C']:
					incoming = []
					for k in self.graph.predecessors(j):
							# This logic doesn't work in our case
							# L = self.graph.edges[(j,k)]['L'] #extract lead time

							# if t - L >= 0: # check if delivery has arrived
							# 	delivery = self.R.loc[t-L,(k,j)]
							# else:
							# 	delivery = np.zeros(self.num_products)

							delivery = self.R.loc[t,(k,j)]
							incoming += [delivery]
							# self.Y.loc[t+1,(k,j)] = [element1 - element2 for (element1, element2) in zip(self.Y.loc[t,(k,j)], delivery )]
							# self.Y.loc[t+1,(k,j)] = [sum(x) for x in zip(self.Y.loc[t+1,(k,j)], self.R.loc[t,(k,j)])]

					incoming = np.array(incoming).reshape(self.num_products, -1)

			else:
				print("you didn't update me")
				 # FILL THIS OUT

			outgoing = np.array(outgoing).reshape(self.num_products, -1) #consumed inventory (for requests placed)
			self.X.loc[t+1,j] = np.array(self.X.loc[t,j]) + np.sum(incoming, axis=1) - np.sum(outgoing, axis=1)
			

		for j in self.retail:
			for k in self.market:
				if (j, k) in self.graph.edges() or (k, j) in self.graph.edges():
					Demand = self.graph.edges[(j,k)]['user_D']
					self.D.loc[t,(j,k)] = Demand[t]
					
					d = self.D.loc[t,(j,k)]
					#satisfy demand up to available level
					X_retail = self.X.iloc[t+1,j] #get inventory at retail before demand was realized
					for i in range(self.num_products):
						self.S.loc[t,(j,k)][i] = min(d[i], X_retail[i])
				# 		print("X before update: ", self.X.loc[t+1,j][i])
						self.X.loc[t+1,j][i] -= self.S.loc[t,(j,k)][i] #update inventory
				# 		print("X after update: ", self.X.loc[t+1,j][i])
						self.U.loc[t,(j,k)][i] = d[i] - self.S.loc[t,(j,k)][i] #update unfulfilled orders


		# calculate profit
	
		'''
		I will temporarily set this to compute just the cost of 
		unmet demand and the unit transportation cost
		'''
		for j in self.main_nodes:
			a = self.alpha
			# SR, PC, HC, OC, = 0,0,0,0
			for i in range(self.num_products):
				# commented out sales revenue and holding cost for now
				# SR += np.sum([self.graph.edges[(j,k)]['p'] * int(self.S.loc[t,(j,k)][i]) for k in self.graph.successors(j)]) #sales revenue
				PC = np.sum([abs(self.graph.edges[(k,j)]['p']) * abs(self.R.loc[t,(k,j)][i]) for k in self.graph.predecessors(j)]) #purchasing c(transport cost)
				if PC == 0:
				    PC = 10
				# if PC<0:
				#     for k in  self.graph.predecessors(j):
				#         print("p: ", self.graph.edges[(k,j)]['p'], "r: ",  self.R.loc[t,(k,j)][i] )
				# if t==2:
				#     for k in self.graph.predecessors(j):
				#         print("p val: {}".format(self.graph.edges[(k,j)]['p']), '\n', "R Val: {}".format(self.R.loc[t,(k,j)][i]))
				# if 'h' in self.graph.nodes[j]:
					# HC += self.graph.nodes[j]['h'] * self.X.loc[t+1,j][i] + np.sum([self.graph.edges[(k,j)]['g'] * self.Y.loc[t+1,(k,j)][i] for k in self.graph.predecessors(j)]) #holding costs

			if j in self.retail:
				UP = np.sum([self.graph.edges[(j,k)]['b'] * self.U.loc[t,(j,k)] for k in self.graph.successors(j)]) #unfulfilled penalty
				for k in self.graph.successors(j):
				    if UP<0:
				        print("unfulfilled: {}".format(self.U.loc[t,(j,k)]))
			else:
				UP = 0
			# self.P.loc[t,j] = a**t * (SR - PC - HC - UP)
			self.P.loc[t,j] = a**t * (-1*PC - UP)

		# update period
		self.period += 1

		# set reward (profit from current timestep) -- cost in our case
		reward = self.P.loc[t,:].sum()

		# determine if simulation should terminate
		if self.period >= self.max_episode_steps:
				done = True
		else:
				done = False
				# update state
				self._update_state()
# 		if t==2:
#             for k in self.graph.predecessors(j):
#                 print("p val: {}".format(self.graph.edges[(k,j)]['p'] ), '\n', "Repl: {}".format(self.R.loc[t,(k,j)][i]),'\n')
        
		print("Number of episodes:", self.period)
		print("Total episodes:", self.max_episode_steps)
		print("Reward:", reward)
		print("Done:", done)
# 		print("Purchasing cost t={}: ".format(t) , PC)
# 		print("P Value: {}".format(self.P.loc[t,j]))
		print("----------------")

		return self.state, reward, done, {}
	
	def sample_action(self):
		'''
		Generate an action by sampling from the action_space
		'''
		sample = self.action_space.sample()
		if all(value > 0 for value in sample.values()):
			return sample
		else:
			sample = self.action_space.sample()

	def step(self, action):
		return self._STEP(action)

	def reset(self):
		return self._RESET()